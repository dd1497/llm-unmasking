# author: ddukic

import json

from constants import *
from datasets import load_dataset
from torch.utils import data

# set by roberta limit
# MAX_LEN = 512

# fighting OOM issues
MAX_LEN = 128


class TokenClassificationDataset(data.Dataset):
    def __init__(self, dataset_name, tokenizer, split):
        self.tokenizer = tokenizer
        self.tokens, self.labels, self.label_ids = [], [], []
        self.dataset_name = dataset_name
        self.split = split
        self.all_labels, self.id2label, self.label2id = self.get_labels(dataset_name)

        self.load_data_split(dataset_name, split)

    def build_vocab(self, labels):
        # OTHER token has idx 0
        all_labels = ["O"]
        for label in labels:
            all_labels.append("B-{}".format(label.lower()))
            all_labels.append("I-{}".format(label.lower()))
        id2label = {id: tag for id, tag in enumerate(all_labels)}
        label2id = {tag: id for id, tag in enumerate(all_labels)}

        return all_labels, id2label, label2id

    def get_labels(self, dataset_name):
        if dataset_name == "maven-ti" or dataset_name == "ace-ti":
            all_labels, id2label, label2id = self.build_vocab(["trigger"])
        elif dataset_name == "maven-tc":
            all_labels, id2label, label2id = self.build_vocab(MAVEN_TRIGGERS)
        elif dataset_name == "ace-tc":
            all_labels, id2label, label2id = self.build_vocab(ACE_TRIGGERS)
        elif dataset_name == "absa-restaurants":
            all_labels, id2label, label2id = self.build_vocab(
                ["positive", "negative", "neutral", "conflict"]
            )
        elif dataset_name == "conll2003":
            label2id = {
                "O": 0,
                "B-PER": 1,
                "I-PER": 2,
                "B-ORG": 3,
                "I-ORG": 4,
                "B-LOC": 5,
                "I-LOC": 6,
                "B-MISC": 7,
                "I-MISC": 8,
            }

            id2label = {v: k for k, v in label2id.items()}

            all_labels = list(label2id.keys())
        elif dataset_name == "conll2003pos":
            label2id = {
                '"': 0,
                "''": 1,
                "#": 2,
                "$": 3,
                "(": 4,
                ")": 5,
                ",": 6,
                ".": 7,
                ":": 8,
                "``": 9,
                "CC": 10,
                "CD": 11,
                "DT": 12,
                "EX": 13,
                "FW": 14,
                "IN": 15,
                "JJ": 16,
                "JJR": 17,
                "JJS": 18,
                "LS": 19,
                "MD": 20,
                "NN": 21,
                "NNP": 22,
                "NNPS": 23,
                "NNS": 24,
                "NN|SYM": 25,
                "PDT": 26,
                "POS": 27,
                "PRP": 28,
                "PRP$": 29,
                "RB": 30,
                "RBR": 31,
                "RBS": 32,
                "RP": 33,
                "SYM": 34,
                "TO": 35,
                "UH": 36,
                "VB": 37,
                "VBD": 38,
                "VBG": 39,
                "VBN": 40,
                "VBP": 41,
                "VBZ": 42,
                "WDT": 43,
                "WP": 44,
                "WP$": 45,
                "WRB": 46,
            }

            id2label = {v: k for k, v in label2id.items()}

            all_labels = list(label2id.keys())
        elif dataset_name == "conll2003chunk":
            label2id = {
                "O": 0,
                "B-ADJP": 1,
                "I-ADJP": 2,
                "B-ADVP": 3,
                "I-ADVP": 4,
                "B-CONJP": 5,
                "I-CONJP": 6,
                "B-INTJ": 7,
                "I-INTJ": 8,
                "B-LST": 9,
                "I-LST": 10,
                "B-NP": 11,
                "I-NP": 12,
                "B-PP": 13,
                "I-PP": 14,
                "B-PRT": 15,
                "I-PRT": 16,
                "B-SBAR": 17,
                "I-SBAR": 18,
                "B-UCP": 19,
                "I-UCP": 20,
                "B-VP": 21,
                "I-VP": 22,
            }

            id2label = {v: k for k, v in label2id.items()}

            all_labels = list(label2id.keys())
        return all_labels, id2label, label2id

    def load_conll2003(self, dataset_name, split):
        data_split = load_dataset("conll2003", split=split)

        self.tokens = data_split["tokens"]
        if dataset_name == "conll2003":
            self.label_ids = data_split["ner_tags"]
        elif dataset_name == "conll2003pos":
            self.label_ids = data_split["pos_tags"]
        elif dataset_name == "conll2003chunk":
            self.label_ids = data_split["chunk_tags"]
        for l in self.label_ids:
            self.labels.append([self.id2label[t] for t in l])

    def load_maven(self, dataset_name, split):
        if split == "train":
            fpath = REPO_HOME + "data/processed/maven/train.jsonl"
        elif split == "validation":
            fpath = REPO_HOME + "data/processed/maven/valid.jsonl"
        else:
            # split must be test
            fpath = REPO_HOME + "data/processed/maven/test.jsonl"

        with open(fpath, "r", encoding="utf-8") as f:
            data = [json.loads(ln) for ln in f.readlines()]
            for item in data:
                toks = [cnt["tokens"] for cnt in item["content"]]

                for sent_id, token_sent in enumerate(toks):
                    tags = ["O"] * len(token_sent)

                    for event_item in item["events"]:
                        if dataset_name == "maven-ti":
                            trigger_type = "trigger"
                        else:
                            trigger_type = event_item["type"]
                        for event_mention in event_item["mention"]:
                            if event_mention["sent_id"] == sent_id:
                                for i in range(
                                    event_mention["offset"][0],
                                    event_mention["offset"][1],
                                ):
                                    if i == event_mention["offset"][0]:
                                        tags[i] = "B-{}".format(trigger_type.lower())
                                    else:
                                        tags[i] = "I-{}".format(trigger_type.lower())

                    self.tokens.append(token_sent)
                    self.labels.append(tags)
                    self.label_ids.append([self.label2id[x] for x in tags])

    def load_ace(self, dataset_name, split):
        if split == "train":
            fpath = REPO_HOME + "data/processed/ace/train.json"
        elif split == "validation":
            fpath = REPO_HOME + "data/processed/ace/dev.json"
        else:
            # split must be test
            fpath = REPO_HOME + "data/processed/ace/test.json"

        with open(fpath, "r") as f:
            data = json.load(f)
            for item in data:
                words = item["words"]
                tags = ["O"] * len(words)

                for event_mention in item["golden-event-mentions"]:
                    for i in range(
                        event_mention["trigger"]["start"],
                        event_mention["trigger"]["end"],
                    ):
                        if dataset_name == "ace-ti":
                            trigger_type = "trigger"
                        else:
                            trigger_type = event_mention["event_type"]
                        if i == event_mention["trigger"]["start"]:
                            tags[i] = "B-{}".format(trigger_type.lower())
                        else:
                            tags[i] = "I-{}".format(trigger_type.lower())

                self.tokens.append(words)
                self.labels.append(tags)
                self.label_ids.append([self.label2id[x] for x in tags])

    def load_absa(self, dataset_name, split):
        from ast import literal_eval

        import pandas as pd

        if split == "train":
            fpath = REPO_HOME + "data/processed/" + dataset_name + "/train.csv"
        elif split == "validation":
            fpath = REPO_HOME + "data/processed/" + dataset_name + "/valid.csv"
        else:
            # split must be test
            fpath = REPO_HOME + "data/processed/" + dataset_name + "/test.csv"

        df = pd.read_csv(fpath)

        df["tokens"] = df["tokens"].apply(literal_eval)
        df["tags"] = df["tags"].apply(literal_eval)

        self.tokens = df.tokens.tolist()
        self.labels = df.tags.tolist()
        for x in self.labels:
            self.label_ids.append([self.label2id[y] for y in x])

    def load_data_split(self, dataset_name, split):
        if "conll2003" in dataset_name:
            self.load_conll2003(dataset_name, split)
        elif "maven" in dataset_name:
            self.load_maven(dataset_name, split)
        elif "ace" in dataset_name:
            self.load_ace(dataset_name, split)
        elif "absa" in dataset_name:
            self.load_absa(dataset_name, split)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens_idx, labels_idx = self.tokens[idx], self.label_ids[idx]

        example_tokenized = self.tokenize_and_align_labels(tokens_idx, labels_idx)

        return example_tokenized

    def tokenize_and_align_labels(self, tokens, tags_id):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        tokenized_input = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True, max_length=MAX_LEN
        )

        word_ids = tokenized_input.word_ids(
            batch_index=0
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(tags_id[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        tokenized_input["labels"] = label_ids
        return tokenized_input
