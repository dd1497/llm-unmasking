# author: ddukic

import argparse
import os
import random

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from constants import REPO_HOME
from dataset import *
from lm_dataset import LMDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler
from trainer import Trainer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

os.environ["WANDB_MODE"] = "offline"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, help="Learning rate for optimizer")
parser.add_argument(
    "--model_id",
    type=str,
    help="Hugging face model identifier",
)
parser.add_argument(
    "--do_lower_case",
    type=parse_boolean,
    help="Transformer tokenizer option for lowercasing",
)
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--batch_size", type=int, help="Size of a batch")
parser.add_argument(
    "--model_save_path",
    type=str,
    help="For example models/maven-ti/roberta",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Random seed",
)
parser.add_argument(
    "--name",
    type=str,
    help="Name of experiment on W&B",
)
parser.add_argument(
    "--is_decoder",
    type=parse_boolean,
    default=False,
    help="Use encoder as decoder model",
)

args = parser.parse_args()

# gotta count add_argument calls
all_passed = sum([v is not None for k, v in vars(args).items()]) == len(vars(args))

print("All arguments passed?", all_passed)

if not all_passed:
    exit(1)

run = wandb.init(
    project="generative-ie-paper",
    entity="ddukic",
    name=args.name,
    config=args,
    mode="offline",
    dir=REPO_HOME,
)

config = run.config

print(config.name)

run.define_metric("epoch")
run.define_metric("train/epoch*", step_metric="epoch")

if __name__ == "__main__":
    accelerator = Accelerator(gradient_accumulation_steps=8)

    print("Accelerator mixed precision:", accelerator.mixed_precision)

    set_seed(config.seed)

    dataset_train = LMDataset(causal=config.is_decoder)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, do_lower=config.do_lower_case, add_prefix_space=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False if config.is_decoder else 0.15
    )

    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        sampler=RandomSampler(
            dataset_train, generator=torch.Generator().manual_seed(config.seed)
        ),
        # doesn't have to do anything with GPU memory
        num_workers=8,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    config_model = AutoConfig.from_pretrained(
        config.model_id,
        num_hidden_layers=4,
        classifier_dropout=0.1,
        is_decoder=config.is_decoder,
    )

    if config.is_decoder:
        model_final = AutoModelForCausalLM.from_config(
            config=config_model,
        )
    else:
        model_final = AutoModelForMaskedLM.from_config(
            config=config_model,
        )

    optimizer = AdamW(
        model_final.parameters(),
        betas=(0.9, 0.95),
        eps=1e-5,
        lr=config.lr,
        weight_decay=0.1,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=len(loader_train) * config.epochs)

    # pass in any order
    (
        model_final,
        optimizer,
        loader_train,
        scheduler,
    ) = accelerator.prepare(model_final, optimizer, loader_train, scheduler)

    trainer = Trainer(
        model=model_final,
        loader=loader_train,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        seed=config.seed,
        epochs=config.epochs,
        run=run,
        MODEL_SAVE_PATH=config.model_save_path,
        # save 5 times in each epoch
        save_steps=len(loader_train) // 5,
    )

    trainer.train()

    run.finish()
