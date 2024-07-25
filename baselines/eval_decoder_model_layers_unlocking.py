# author: ddukic

import argparse
import os
from collections import defaultdict
from statistics import mean, stdev

import wandb
from constants import REPO_HOME
from trainer_eval_utils_layers_unlocking import *

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
    "--task_type",
    type=str,
    help="Can be 'token_clf' or 'seq_clf' or 'seq_as_token_clf'",
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
    loader = prepare_data(config, split=config.split)

    unlock_configs = generate_unlock_configurations()

    chkpts = os.listdir(os.path.join(REPO_HOME, config.models_saved_path))

    for id, unlock_config in enumerate(unlock_configs):
        chkpts_config = [x for x in chkpts if int(x.split("-")[-1]) == id]

        print("Unlock config: ", unlock_config)
        print("Filtered chkpts:", chkpts_config)

        results = defaultdict(list)

        for chkpt_name in chkpts_config:
            model_final = load_quantized_peft_decoder_model(
                config=config,
                num_labels=len(loader.dataset.all_labels),
                id2label=loader.dataset.id2label,
                label2id=loader.dataset.label2id,
                chkpt_name=chkpt_name,
            )

            if config.task_type == "token_clf":
                metrics = compute_metrics_token_clf_decoder_model(
                    model_final, loader, config, unlock_config
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

        run.summary[
            config.split + "_all_metrics" + "_unlock_config_" + str(id)
        ] = results

    run.finish()
