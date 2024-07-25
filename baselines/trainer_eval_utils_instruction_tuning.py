# author: ddukic

import os
import random

import evaluate
import numpy as np
import torch
import torch.nn as nn
from adapter_config import *
from bitsandbytes.optim import GlobalOptimManager, PagedAdamW
from constants import REPO_HOME
from dataset import *
from dataset_instruction_tuning import *
from datasets import Dataset
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    GenerationConfig,
)
from transformers import set_seed as transformers_set_seed
from transformers.trainer_pt_utils import get_parameter_names

seqeval = evaluate.load("seqeval")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_data(config, split, seed=None):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, do_lower=config.do_lower_case, add_prefix_space=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    dataset = InstructionTuningDatasetTokenClassification(
        dataset_name=config.dataset_name, split=split
    )

    if split == "train":
        dataset_final = Dataset.from_list(dataset)
    else:
        dataset_final = dataset

    return dataset_final, tokenizer


def prepare_training_necessities(
    config,
    model,
    T_max,
):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.1,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    # dealing with memory spikes when training with qlora
    optimizer = PagedAdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.95),
        eps=1e-5,
        lr=config.lr,
        weight_decay=0.1,
        optim_bits=8,
    )

    # from this issue https://github.com/huggingface/transformers/issues/14819
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            GlobalOptimManager.get_instance().register_module_override(
                module, "weight", {"optim_bits": 32}
            )

    scheduler = CosineAnnealingLR(optimizer, T_max=T_max * config.epochs)

    return optimizer, scheduler


def format_instruction_train(sample):
    return f"""### Instruction: 
{sample["instruction"]} 

### Options:
{sample["options"]}

### Sentence:
{sample["sentence"]}

### Response:
{sample["output"]}
"""


def format_instruction_eval(sample):
    return f"""### Instruction: 
{sample["instruction"]} 

### Options:
{sample["options"]}

### Sentence:
{sample["sentence"]}

### Response:
"""


def get_quantized_peft_decoder_model(config):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, quantization_config=nf4_config, device_map="auto"
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    model_final = get_peft_model(model, peft_lora_config_clm_decoder)

    return model_final


def get_quantized_peft_encoder_decoder_model(config):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_id, quantization_config=nf4_config, device_map="auto"
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    model_final = get_peft_model(model, peft_lora_config_seq2seq)

    return model_final


def load_quantized_peft_decoder_model(config, chkpt_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=nf4_config,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model_final = PeftModel.from_pretrained(
        model=model,
        model_id=os.path.join(REPO_HOME, config.models_saved_path, chkpt_name),
        device_map="auto",
    )

    return model_final


class MyCollatorInstructionTuning(DataCollatorWithPadding):
    def __call__(self, features):
        prompt = [format_instruction_eval(feature) for feature in features]
        tokenized_prompts = self.tokenizer(prompt)

        batch = self.tokenizer.pad(
            tokenized_prompts,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch["labels"] = [x["labels"] for x in features]
        batch["tokens"] = [x["tokens"] for x in features]

        return batch


def compute_metrics_instruction_tuned_decoder_model(
    model, dataset, tokenizer, chkpt_name, config
):
    # for batched generation I need left padding
    if tokenizer.padding_side == "right":
        tokenizer.padding_side = "left"
    # also for batched generation I need to call model.bfloat16() although I already loaded model in bfloat16
    if config.base_model_name == "llama" or "t5":
        model = model.bfloat16()

    data_collator = MyCollatorInstructionTuning(tokenizer, padding="longest")

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        sampler=None,
        # doesn't have to do anything with GPU memory
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    transformers_set_seed(int(chkpt_name.split("-")[1]))

    model.eval()

    predictions = []
    targets = []

    with torch.inference_mode():
        i = 0

        for batch in tqdm(loader):
            # have same generation for llama and mistral
            out = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                generation_config=GenerationConfig(
                    **{
                        "bos_token_id": 1,
                        "do_sample": True,
                        "eos_token_id": 2,
                        "max_length": 1024,
                        "pad_token_id": tokenizer.eos_token_id,
                        "temperature": 0.6,
                        "top_p": 0.9,
                    },
                ),
            )

            for prompt, response, labels, tokens in zip(
                tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True),
                tokenizer.batch_decode(out, skip_special_tokens=True),
                batch["labels"],
                batch["tokens"],
            ):
                if config.base_model_name != "t5":
                    spans = response[len(prompt) :]
                else:
                    spans = response

                try:
                    pred = dataset.spans_to_bio_tags(spans, tokens)
                    target = labels
                    predictions.append(pred)
                    targets.append(target)
                except Exception as e:
                    print("Failed for index ", str(i), " with exception: ", e)
                    pred = ["O"] * len(tokens)

                    target = labels
                    predictions.append(pred)
                    targets.append(target)

                i += 1

    # must use IOB2 for evaluation
    all_metrics_classification = seqeval.compute(
        predictions=predictions, references=targets, scheme="IOB2", mode="strict"
    )

    pruned_metrics = {
        "f1": all_metrics_classification["overall_f1"],
        "p": all_metrics_classification["overall_precision"],
        "r": all_metrics_classification["overall_recall"],
        "acc": all_metrics_classification["overall_accuracy"],
    }

    print(pruned_metrics)

    return pruned_metrics
