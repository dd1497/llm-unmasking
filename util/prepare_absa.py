# author: ddukic

import random
import xml.etree.ElementTree as ET

import pandas as pd
import spacy

# Your XML data

nlp = spacy.load("en_core_web_lg")


def prepare_absa(xml_file_path):
    tree = ET.parse(xml_file_path)

    root = tree.getroot()

    tokens_all = []
    tags_all = []

    for sentence in root.findall(".//sentence"):
        sentence_id = sentence.get("id")
        text = sentence.find("text").text

        terms = []

        for aspect_term in sentence.findall(".//aspectTerm"):
            terms.append(
                {
                    "polarity": aspect_term.get("polarity"),
                    "from": int(aspect_term.get("from")),
                    "to": int(aspect_term.get("to")),
                }
            )

        tokens = nlp(text)
        bio_tags = ["O"] * len(tokens)
        tokens_text = []

        for i in range(len(terms)):
            label = terms[i]["polarity"]
            args_start = terms[i]["from"]
            args_end = terms[i]["to"]
            out = tokens.char_span(args_start, args_end)
            if out is not None:
                start_tok, end_tok = out.start, out.end
                if start_tok == end_tok:
                    bio_tags[start_tok] = "B-" + label
                else:
                    bio_tags[start_tok] = "B-" + label
                    for j in range(start_tok + 1, end_tok):
                        bio_tags[j] = "I-" + label
            else:
                print(sentence_id + " Error with match of " + str(terms[i]) + "\n")

        for t in tokens:
            tokens_text.append(t.text)

        tokens_all.append(tokens_text)
        tags_all.append(bio_tags)

    return tokens_all, tags_all


random.seed(42)

train_tokens, train_tags = prepare_absa("../data/raw/absa/Restaurants_Train_v2.xml")
test_tokens_final, test_tags_final = prepare_absa(
    "../data/raw/absa/Restaurants_Test_Gold.xml"
)

indices_valid = random.sample(
    list(range(len(train_tokens))), int(len(train_tokens) * 0.1)
)

train_tokens_final, train_tags_final = [], []
valid_tokens_final, valid_tags_final = [], []

for i in range(len(train_tokens)):
    if i in indices_valid:
        valid_tokens_final.append(train_tokens[i])
        valid_tags_final.append(train_tags[i])
    else:
        train_tokens_final.append(train_tokens[i])
        train_tags_final.append(train_tags[i])

df_train = pd.DataFrame({"tokens": train_tokens_final, "tags": train_tags_final})
df_valid = pd.DataFrame({"tokens": valid_tokens_final, "tags": valid_tags_final})
df_test = pd.DataFrame({"tokens": test_tokens_final, "tags": test_tags_final})

df_train.to_csv("../data/processed/absa/train.csv", index=False)
df_valid.to_csv("../data/processed/absa/valid.csv", index=False)
df_test.to_csv("../data/processed/absa/test.csv", index=False)
