# author: ddukic

import argparse
import os
from collections import defaultdict
from statistics import mean, stdev

import wandb
from constants import REPO_HOME
from dataset import *
from trainer_eval_utils_instruction_tuning import *

os.environ["WANDB_MODE"] = "offline"


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


parser = argparse.ArgumentParser()
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
    "--models_saved_path",
    type=str,
    help="For example models/maven-ti/llama, where the checkpoints were saved",
)
parser.add_argument(
    "--split", type=str, help="Can be either 'train', 'validation', or 'test'"
)
parser.add_argument(
    "--name",
    type=str,
    help="Name of experiment on W&B",
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

if __name__ == "__main__":
    dataset, tokenizer = prepare_data(config, split=config.split)

    chkpts = os.listdir(os.path.join(REPO_HOME, config.models_saved_path))

    print(chkpts)

    results = defaultdict(list)

    for chkpt_name in chkpts:
        model_final = load_quantized_peft_decoder_model(
            config=config,
            chkpt_name=chkpt_name,
        )

        metrics = compute_metrics_instruction_tuned_decoder_model(
            model_final, dataset, tokenizer, chkpt_name, config
        )

        for k in metrics.keys():
            results[k + "_avg"].append(metrics[k])
            results[k + "_stdev"].append(metrics[k])

    for k in results.keys():
        if "avg" in k:
            results[k] = mean(results[k])
        elif "stdev" in k:
            results[k] = stdev(results[k])

    print(results)

    run.summary[config.split + "_all_metrics"] = results

    run.finish()
