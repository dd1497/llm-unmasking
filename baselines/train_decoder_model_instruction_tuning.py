# author: ddukic

import argparse
import os

import wandb
from constants import REPO_HOME
from trainer_eval_utils_instruction_tuning import *
from transformers import TrainingArguments
from trl import SFTTrainer

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
    "--seeds",
    nargs="+",
    help="Random seeds for each of the training runs, number of training runs is controlled by length of seeds list",
)
parser.add_argument(
    "--name",
    type=str,
    help="Name of experiment on W&B",
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

if __name__ == "__main__":
    for seed in config.seeds:
        set_seed(seed)

        dataset_train, tokenizer = prepare_data(config, "train", seed)

        if "t5" in config.model_id:
            model_final = get_quantized_peft_encoder_decoder_model(config)
        else:
            model_final = get_quantized_peft_decoder_model(config)

        print("=" * 30)
        model_final.print_trainable_parameters()
        print("=" * 30)

        optimizer, scheduler = prepare_training_necessities(
            config=config,
            model=model_final,
            T_max=len(dataset_train) // config.batch_size + 1,
        )

        args = TrainingArguments(
            output_dir=config.model_save_path,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=config.lr,
            bf16=True,
            save_strategy="no",
            max_grad_norm=1.0,
            seed=seed,
            report_to="wandb",
            logging_strategy="epoch",
            logging_first_step=True,
            overwrite_output_dir=True,
        )

        # for conll2003 with instructions, no examples are longer than 350
        # for absa restaurants, no examples are longer than 250
        max_seq_length = (
            1024  # max sequence length for model and packing of the dataset
        )

        if "absa" in config.dataset_name:
            max_seq_length = 512

        trainer = SFTTrainer(
            model=model_final,
            train_dataset=dataset_train,
            max_seq_length=max_seq_length,
            optimizers=(optimizer, scheduler),
            tokenizer=tokenizer,
            formatting_func=format_instruction_train,
            args=args,
            packing=True,
        )

        trainer.train()

        # save only last model
        trainer.save_model(
            os.path.join(
                REPO_HOME,
                config.model_save_path,
                "seed-" + str(seed) + "-epoch-" + str(config.epochs - 1) + "-step-"
                # 4 gradient accumulation steps
                # https://stackoverflow.com/questions/76002567/how-is-the-number-of-steps-calculated-in-huggingface-trainer
                + str(int(len(dataset_train) / config.batch_size // 4)),
            )
        )

    run.finish()
