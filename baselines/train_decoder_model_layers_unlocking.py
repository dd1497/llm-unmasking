# author: ddukic

import argparse
import gc
import os

import torch
import wandb
from accelerate import Accelerator
from constants import REPO_HOME
from dataset import *
from trainer_eval_utils_layers_unlocking import *
from trainer_layers_unlocking import Trainer

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
    help="Internal model name such as 'llama' or 'opt' or 'mistral'",
)
parser.add_argument(
    "--model_save_path",
    type=str,
    help="For example models/maven-ti/llama",
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
    "--tapt",
    type=str,
    help="Can be 'clm', 'from-tapt-clm-config-x' or 'no'",
    default="no",
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

if __name__ == "__main__":
    unlock_configs = generate_unlock_configurations()

    for id, unlock_config in enumerate(unlock_configs):
        print("Unlock config: ", unlock_config)
        for seed in config.seeds:
            set_seed(seed)

            if config.tapt == "clm":
                loader_tapt = prepare_data_tapt(config, "train", seed)

                model_tapt = get_quantized_peft_tapt_decoder_model(config)

                print("=" * 30)
                model_tapt.print_trainable_parameters()
                print("=" * 30)

                accelerator_tapt = Accelerator(gradient_accumulation_steps=4)

                print(
                    "Accelerator tapt mixed precision:",
                    accelerator_tapt.mixed_precision,
                )

                optimizer_tapt, scheduler_tapt = prepare_training_necessities(
                    config=config, model=model_tapt, T_max=len(loader_tapt)
                )

                # pass in any order
                (
                    model_tapt,
                    optimizer_tapt,
                    loader_tapt,
                    scheduler_tapt,
                ) = accelerator_tapt.prepare(
                    model_tapt, optimizer_tapt, loader_tapt, scheduler_tapt
                )

                tapt_save_path = config.model_save_path.replace("from-tapt-clm", "tapt")

                trainer_tapt = Trainer(
                    model=model_tapt,
                    loader=loader_tapt,
                    optimizer=optimizer_tapt,
                    scheduler=scheduler_tapt,
                    accelerator=accelerator_tapt,
                    seed=seed,
                    epochs=config.epochs,
                    run=run,
                    MODEL_SAVE_PATH=tapt_save_path,
                    unlock_config=unlock_config,
                    unlock_config_id=id,
                    tapt="_tapt",
                )

                saved_tapt_model_path = trainer_tapt.train()

                model_tapt = model_tapt.to("cpu")
                del optimizer_tapt
                del scheduler_tapt
                del accelerator_tapt
                del loader_tapt
                del trainer_tapt
                del model_tapt

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                print("=" * 30)
                print("TAPT FINISHED")
                print("=" * 30)

            accelerator = Accelerator(gradient_accumulation_steps=4)

            print("Accelerator mixed precision:", accelerator.mixed_precision)

            loader_train = prepare_data(config, "train", seed)

            if config.tapt == "clm":
                model_final = load_quantized_peft_tapt_decoder_model(
                    config=config,
                    num_labels=len(loader_train.dataset.all_labels),
                    id2label=loader_train.dataset.id2label,
                    label2id=loader_train.dataset.label2id,
                    chkpt_path=saved_tapt_model_path,
                )
            elif "from-tapt-clm-config-0" in config.tapt:
                # pick from which tapt checkpoint to train

                chkpts = os.listdir(
                    os.path.join(
                        REPO_HOME,
                        "models/" + config.dataset_name,
                        (
                            "llama2-7b-layers-unlocking-tapt"
                            if config.base_model_name == "llama"
                            else "mistral-7b-layers-unlocking-tapt"
                        ),
                    )
                )

                # for now train always from checkpoint 0
                chkpt_config_tapt = [
                    x
                    for x in chkpts
                    if int(x.split("-")[1]) == seed and int(x.split("-")[-1]) == 0
                ][0]

                print(
                    os.path.join(
                        REPO_HOME,
                        "models/" + config.dataset_name,
                        (
                            "llama2-7b-layers-unlocking-tapt"
                            if config.base_model_name == "llama"
                            else "mistral-7b-layers-unlocking-tapt"
                        ),
                        chkpt_config_tapt,
                    )
                )

                model_final = load_quantized_peft_tapt_decoder_model(
                    config=config,
                    num_labels=len(loader_train.dataset.all_labels),
                    id2label=loader_train.dataset.id2label,
                    label2id=loader_train.dataset.label2id,
                    chkpt_path=os.path.join(
                        REPO_HOME,
                        "models/" + config.dataset_name,
                        (
                            "llama2-7b-layers-unlocking-tapt"
                            if config.base_model_name == "llama"
                            else "mistral-7b-layers-unlocking-tapt"
                        ),
                        chkpt_config_tapt,
                    ),
                )
            else:
                model_final = get_quantized_peft_decoder_model(
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
                unlock_config=unlock_config,
                unlock_config_id=id,
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
