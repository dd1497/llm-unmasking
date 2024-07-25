# author: ddukic

import os
import random

import evaluate
import numpy as np
import torch
import torch.nn as nn
from adapter_config import *
from bert_model import *
from bitsandbytes.optim import GlobalOptimManager, PagedAdamW
from constants import REPO_HOME
from dataset import *
from llama_model_layers_unlocking import *
from lm_dataset import LMDatasetTAPT
from mistral_model_layers_unlocking import *
from opt_model import *
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from roberta_model import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
)
from transformers.trainer_pt_utils import get_parameter_names

seqeval = evaluate.load("seqeval")
poseval = evaluate.load("poseval")

model_map_seq_clf = {
    "roberta": RobertaForSequenceClassificationLockUnlockAttention,
}

model_map_token_clf = {
    "roberta": RobertaForTokenClassificationLockUnlockAttention,
    "llama": LlamaForTokenClassificationLockUnlockAttention,
    "mistral": MistralForTokenClassificationLockUnlockAttention,
}

model_map_clm = {
    "llama": LlamaForCausalLMLockUnlockAttention,
    "mistral": MistralForCausalLMLockUnlockAttention,
}


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

    if config.task_type == "token_clf":
        dataset = TokenClassificationDataset(
            dataset_name=config.dataset_name, tokenizer=tokenizer, split=split
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer, padding="longest"
        )

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        sampler=(
            RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
            if split == "train"
            else None
        ),
        # doesn't have to do anything with GPU memory
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    return loader


def prepare_data_tapt(config, split, seed=None):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, do_lower=config.do_lower_case, add_prefix_space=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    if config.tapt == "clm":
        dataset = LMDatasetTAPT(
            config.dataset_name, tokenizer, split=split, causal=True
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)),
        # doesn't have to do anything with GPU memory
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    return loader


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


def get_quantized_peft_encoder_model(config, num_labels, id2label, label2id):
    # quantization and mixed precision setup
    if config.task_type == "seq_clf":
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["classifier"],
        )

        model = model_map_seq_clf[config.base_model_name].from_pretrained(
            config.model_id,
            num_labels=num_labels,
            quantization_config=nf4_config,
            is_decoder=config.lock_attention,
        )

        model = prepare_model_for_kbit_training(model)

        model_final = get_peft_model(model, peft_lora_config_seq_clf_encoder)
    else:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = model_map_token_clf[config.base_model_name].from_pretrained(
            config.model_id,
            num_labels=num_labels,
            quantization_config=nf4_config,
            is_decoder=config.lock_attention,
        )

        model = prepare_model_for_kbit_training(model)

        model_final = get_peft_model(model, peft_lora_config_token_clf_encoder)

        model_final.config.id2label = id2label
        model_final.config.label2id = label2id

    return model_final


def load_quantized_peft_encoder_model(
    config, num_labels, id2label, label2id, chkpt_name
):
    # quantization and mixed precision setup
    if config.task_type == "seq_clf":
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["classifier"],
        )

        model = model_map_seq_clf[config.base_model_name].from_pretrained(
            config.model_id,
            num_labels=num_labels,
            quantization_config=nf4_config,
            is_decoder=config.lock_attention,
        )

        model_final = PeftModel.from_pretrained(
            model,
            os.path.join(REPO_HOME, config.models_saved_path, chkpt_name),
            device_map="auto",
        )
    else:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = model_map_token_clf[config.base_model_name].from_pretrained(
            config.model_id,
            num_labels=num_labels,
            quantization_config=nf4_config,
            is_decoder=config.lock_attention,
        )

        model_final = PeftModel.from_pretrained(
            model,
            os.path.join(REPO_HOME, config.models_saved_path, chkpt_name),
            device_map="auto",
        )

        model_final.config.id2label = id2label
        model_final.config.label2id = label2id

    return model_final


def get_quantized_peft_tapt_decoder_model(config):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if config.tapt == "clm":
        model = model_map_clm[config.base_model_name].from_pretrained(
            config.model_id,
            quantization_config=nf4_config,
        )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    if config.tapt == "clm":
        model_final = get_peft_model(model, peft_lora_config_clm_decoder)

    return model_final


def get_quantized_peft_decoder_model(config, num_labels, id2label, label2id):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if config.task_type == "seq_clf":
        model = model_map_seq_clf[config.base_model_name].from_pretrained(
            config.model_id,
            num_labels=num_labels,
            quantization_config=nf4_config,
        )
    else:
        model = model_map_token_clf[config.base_model_name].from_pretrained(
            config.model_id,
            num_labels=num_labels,
            quantization_config=nf4_config,
        )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    if config.task_type == "seq_clf":
        model_final = get_peft_model(model, peft_lora_config_seq_clf_decoder)
    else:
        model_final = get_peft_model(model, peft_lora_config_token_clf_decoder)

        model_final.config.id2label = id2label
        model_final.config.label2id = label2id

    return model_final


def load_quantized_peft_decoder_model(
    config, num_labels, id2label, label2id, chkpt_name
):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if config.task_type == "seq_clf":
        model = model_map_seq_clf[config.base_model_name].from_pretrained(
            config.model_id,
            num_labels=num_labels,
            quantization_config=nf4_config,
        )
    else:
        model = model_map_token_clf[config.base_model_name].from_pretrained(
            config.model_id,
            num_labels=num_labels,
            quantization_config=nf4_config,
        )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    if config.task_type == "seq_clf":
        model_final = PeftModel.from_pretrained(
            model,
            os.path.join(REPO_HOME, config.models_saved_path, chkpt_name),
            device_map="auto",
        )
    else:
        model_final = PeftModel.from_pretrained(
            model,
            os.path.join(REPO_HOME, config.models_saved_path, chkpt_name),
            device_map="auto",
        )

        model_final.config.id2label = id2label
        model_final.config.label2id = label2id

    return model_final


def load_quantized_peft_tapt_decoder_model(
    config, num_labels, id2label, label2id, chkpt_path
):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = model_map_token_clf[config.base_model_name].from_pretrained(
        config.model_id,
        num_labels=num_labels,
        quantization_config=nf4_config,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model_final = PeftModel.from_pretrained(
        model=model,
        model_id=os.path.join(REPO_HOME, chkpt_path),
        # really important!!
        is_trainable=True,
        config=peft_lora_config_token_clf_decoder,
        device_map="auto",
    )

    model_final.config.id2label = id2label
    model_final.config.label2id = label2id

    return model_final


def compute_metrics_token_clf_decoder_model(model, loader, config, unlock_config):
    model.eval()

    predictions = []
    targets = []

    model.config.use_cache = True

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch, unlock_config=unlock_config)
            pred = torch.argmax(out.logits, dim=-1)
            target = batch["labels"]
            predictions.extend(pred.tolist())
            targets.extend(target.tolist())

    predictions = [
        [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, targets)
    ]

    targets = [
        [model.config.id2label[l] for l in label if l != -100] for label in targets
    ]

    if config.dataset_name == "conll2003pos":
        # POS is evaluated by treating each token independently
        all_metrics_classification = poseval.compute(
            predictions=predictions, references=targets
        )

        pruned_metrics = {
            "f1": all_metrics_classification["macro avg"]["f1-score"],
            "p": all_metrics_classification["macro avg"]["precision"],
            "r": all_metrics_classification["macro avg"]["recall"],
            "acc": all_metrics_classification["accuracy"],
        }
    else:
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


def compute_metrics_token_clf_encoder_model(model, loader, config):
    model.eval()

    predictions = []
    targets = []

    model.config.use_cache = True

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            pred = torch.argmax(out.logits, dim=-1)
            target = batch["labels"]
            predictions.extend(pred.tolist())
            targets.extend(target.tolist())

    predictions = [
        [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, targets)
    ]

    targets = [
        [model.config.id2label[l] for l in label if l != -100] for label in targets
    ]

    if config.dataset_name == "conll2003pos":
        # POS is evaluated by treating each token independently
        all_metrics_classification = poseval.compute(
            predictions=predictions, references=targets
        )

        pruned_metrics = {
            "f1": all_metrics_classification["macro avg"]["f1-score"],
            "p": all_metrics_classification["macro avg"]["precision"],
            "r": all_metrics_classification["macro avg"]["recall"],
            "acc": all_metrics_classification["accuracy"],
        }
    else:
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


def generate_unlock_configurations(layers=32, group_layer_by=8):
    from itertools import product

    indices = []
    layers_breaks = list(range(0, layers + 1, group_layer_by))
    for i in range(len(layers_breaks)):
        if i != len(layers_breaks) - 1:
            indices.append(list(range(layers_breaks[i], layers_breaks[i + 1])))

    unlock_configurations = [[False] * layers for _ in range(2 ** len(indices))]

    bit_sets = product(range(2), repeat=len(indices))

    for i, bset in enumerate(bit_sets):
        for j in range(len(bset)):
            for layer_idx_in_group in indices[j]:
                unlock_configurations[i][layer_idx_in_group] = bool(bset[j])

    return unlock_configurations
