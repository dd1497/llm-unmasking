# author: ddukic

import json

from constants import *
from datasets import load_dataset
from torch.utils import data

REPO_HOME = "/lustre/home/ddukic/generative-ie/"


class InstructionTuningDatasetTokenClassification(data.Dataset):
    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.split = split

        self.load_data_split(dataset_name, split)

    def load_conll2003(self, dataset_name, split):
        data_split = load_dataset("conll2003", split=split)

        self.sentences = [" ".join(x) for x in data_split["tokens"]]

        if dataset_name == "conll2003":
            self.instruction = "please extract named entities and their type from the input sentence, all entity types are in options"
            self.options = "person, location, organization, miscellaneous"

            self.label2id = {
                "O": 0,
                "B-person": 1,
                "I-person": 2,
                "B-organization": 3,
                "I-organization": 4,
                "B-location": 5,
                "I-location": 6,
                "B-miscellaneous": 7,
                "I-miscellaneous": 8,
            }

            self.id2label = {v: k for k, v in self.label2id.items()}

            self.tokens = data_split["tokens"]
            self.labels = [
                [self.id2label[y] for y in x] for x in data_split["ner_tags"]
            ]

            self.responses = [
                self.extract_bio_spans(x, y)
                for x, y in zip(data_split["tokens"], self.labels)
            ]

        elif dataset_name == "conll2003chunk":
            self.instruction = "please extract chunks and their type from the input sentence, all chunk types are in options"
            self.options = "noun phrase, verb phrase, prepositional phrase, adverb phrase, subordinated clause, adjective phrase, particles, conjunction phrase, interjection, list marker, unlike coordinated phrase"

            self.label2id = {
                "O": 0,
                "B-adjective phrase": 1,
                "I-adjective phrase": 2,
                "B-adverb phrase": 3,
                "I-adverb phrase": 4,
                "B-conjunction phrase": 5,
                "I-conjunction phrase": 6,
                "B-interjection": 7,
                "I-interjection": 8,
                "B-list marker": 9,
                "I-list marker": 10,
                "B-noun phrase": 11,
                "I-noun phrase": 12,
                "B-prepositional phrase": 13,
                "I-prepositional phrase": 14,
                "B-particles": 15,
                "I-particles": 16,
                "B-subordinated clause": 17,
                "I-subordinated clause": 18,
                "B-unlike coordinated phrase": 19,
                "I-unlike coordinated phrase": 20,
                "B-verb phrase": 21,
                "I-verb phrase": 22,
            }

            self.id2label = {v: k for k, v in self.label2id.items()}

            self.tokens = data_split["tokens"]
            self.labels = [
                [self.id2label[y] for y in x] for x in data_split["chunk_tags"]
            ]

            self.responses = [
                self.extract_bio_spans(x, y)
                for x, y in zip(data_split["tokens"], self.labels)
            ]

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

        self.sentences = [" ".join(x) for x in df.tokens.tolist()]
        self.tokens = df.tokens.tolist()
        self.labels = df.tags.tolist()

        self.instruction = "please extract aspect terms and their polarity from the input sentence, all polarity types are in options"
        self.options = "positive, negative, neutral, conflict"

        self.responses = [
            self.extract_bio_spans(x, y) for x, y in zip(self.tokens, self.labels)
        ]

    def load_ace(self, dataset_name, split):
        if split == "train":
            fpath = REPO_HOME + "data/processed/ace/train.json"
        elif split == "validation":
            fpath = REPO_HOME + "data/processed/ace/dev.json"
        else:
            # split must be test
            fpath = REPO_HOME + "data/processed/ace/test.json"

        self.tokens, self.labels = [], []

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
                            tags[i] = "B-{}".format(
                                ACE_TRIGGERS_TO_OPTIONS[trigger_type].lower()
                            )
                        else:
                            tags[i] = "I-{}".format(
                                ACE_TRIGGERS_TO_OPTIONS[trigger_type].lower()
                            )
                self.tokens.append(words)
                self.labels.append(tags)

        self.sentences = [" ".join(x) for x in self.tokens]
        self.instruction = "please extract events and their types from the input sentence, all event types are in options"
        self.options = ", ".join([x.lower() for x in ACE_TRIGGERS_TO_OPTIONS.values()])

        self.responses = [
            self.extract_bio_spans(x, y) for x, y in zip(self.tokens, self.labels)
        ]

    def load_data_split(self, dataset_name, split):
        if "conll2003" in dataset_name:
            self.load_conll2003(dataset_name, split)
        elif "absa" in dataset_name:
            self.load_absa(dataset_name, split)
        elif "ace-tc" in dataset_name:
            self.load_ace(dataset_name, split)

    def extract_bio_spans(self, tokens, labels):
        spans = []
        current_span = None
        label_previous = None

        for token, label in zip(tokens, labels):
            if label != "O":
                tag, phrase_type = label.split("-")
                if current_span is not None and (
                    tag == "I" and phrase_type == label_previous
                ):
                    current_span.append((token, phrase_type))
                else:
                    if current_span is not None:
                        spans.append(
                            " ".join([x[0] for x in current_span])
                            + ":"
                            + current_span[0][1]
                        )
                    current_span = [(token, phrase_type)]
                    label_previous = phrase_type
            elif current_span is not None:
                spans.append(
                    " ".join([x[0] for x in current_span]) + ":" + current_span[0][1]
                )
                current_span = None
                label_previous = None

        if current_span is not None:
            spans.append(
                " ".join([x[0] for x in current_span]) + ":" + current_span[0][1]
            )

        return ";".join(spans)

    def spans_to_bio_tags(self, spans, tokens):
        entities = spans.split(";")

        bio_tags = ["O"] * len(tokens)

        if spans.strip() == "":
            return bio_tags

        current_index = 0  # Variable to keep track of the current index

        for entity in entities:
            entity = entity.lstrip().rstrip()
            entity_tokens = [
                x.lstrip().rstrip() for x in entity.rsplit(":", 1)[0].split(" ")
            ]
            entity_type = entity.rsplit(":", 1)[1]

            start_index = current_index
            while (
                start_index < len(tokens)
                and tokens[start_index : start_index + len(entity_tokens)]
                != entity_tokens
            ):
                start_index += 1

            if start_index < len(tokens):
                bio_tags[start_index] = "B-" + entity_type
                for j in range(start_index + 1, start_index + len(entity_tokens)):
                    bio_tags[j] = "I-" + entity_type

                current_index = start_index + len(entity_tokens)

        return bio_tags

    def sanity_check(self):
        for i in range(len(self.sentences)):
            next_ = self.__getitem__(i)
            try:
                if (
                    not self.spans_to_bio_tags(next_["output"], next_["tokens"])
                    == next_["labels"]
                ):
                    print(next_["output"], next_["tokens"], i)
                    print()
            except:
                print("problem with:", i)
                print()

    def format_instruction_train(self, sample):
        return f"""### Instruction: 
{sample["instruction"]} 

### Options:
{sample["options"]}

### Sentence:
{sample["sentence"]}

### Response:
{sample["output"]}
"""

    def format_instruction_eval(self, sample):
        return f"""### Instruction: 
{sample["instruction"]} 

### Options:
{sample["options"]}

### Sentence:
{sample["sentence"]}

### Response:
"""

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "instruction": self.instruction,
            "options": self.options,
            "sentence": self.sentences[idx],
            "output": self.responses[idx],
            "labels": self.labels[idx],
            "tokens": self.tokens[idx],
        }
