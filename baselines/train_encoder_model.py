# author: ddukic

import argparse
import gc
import os

import torch
import wandb
from accelerate import Accelerator
from constants import REPO_HOME
from dataset import *
from trainer import Trainer
from trainer_eval_utils import *

os.environ["WANDB_MODE"] = "offline"


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
    "--dataset_name",
    type=str,
    help="Dataset name depending on the task",
)
parser.add_argument(
    "--base_model_name",
    type=str,
    help="Internal model name such as 'roberta' or 'bert'",
)
parser.add_argument(
    "--model_save_path",
    type=str,
    help="For example models/maven-ti/roberta",
)
parser.add_argument(
    "--task_type",
    type=str,
    help="Can be 'token_clf'",
)
parser.add_argument(
    "--seeds",
    nargs="+",
    help="Random seeds for each of the training runs, number of training runs is controlled by length of seeds list",
)
parser.add_argument(
    "--name",
    type=str,
    help="Name of experiment on W&B",
)
parser.add_argument(
    "--lock_attention",
    type=parse_boolean,
    default=False,
    help="Use encoder as decoder model",
)

args = parser.parse_args()
args.seeds = [int(x) for x in args.seeds]

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
run.define_metric("valid/epoch*", step_metric="epoch")

if __name__ == "__main__":
    for seed in config.seeds:
        accelerator = Accelerator(gradient_accumulation_steps=4)

        print("Accelerator mixed precision:", accelerator.mixed_precision)

        set_seed(seed)

        loader_train = prepare_data(config, "train", seed)

        model_final = get_quantized_peft_encoder_model(
            config=config,
            num_labels=len(loader_train.dataset.all_labels),
            id2label=loader_train.dataset.id2label,
            label2id=loader_train.dataset.label2id,
        )

        print("=" * 30)
        model_final.print_trainable_parameters()
        print("=" * 30)

        optimizer, scheduler = prepare_training_necessities(
            config=config, model=model_final, T_max=len(loader_train)
        )

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
            seed=seed,
            epochs=config.epochs,
            run=run,
            MODEL_SAVE_PATH=config.model_save_path,
            # save 5 times in each epoch
            save_steps=len(loader_train) // 5,
            save_clf_head=True if config.task_type == "seq_clf" else False,
        )

        trainer.train()

        # without this the allocated memory spikes a lot
        model_final = model_final.to("cpu")
        del optimizer
        del scheduler
        del accelerator
        del loader_train
        del trainer
        del model_final

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    run.finish()
