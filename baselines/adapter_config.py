# author: ddukic

from peft import LoraConfig, TaskType

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

peft_lora_config_token_clf_encoder = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type=TaskType.TOKEN_CLS,
    target_modules=["query", "value"],
)

peft_lora_config_token_clf_decoder = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type=TaskType.TOKEN_CLS,
    target_modules=["q_proj", "v_proj"],
)

peft_lora_config_seq_clf_encoder = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value"],
)


peft_lora_config_seq_clf_decoder = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj", "v_proj"],
)

peft_lora_config_clm_decoder = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
)

peft_lora_config_seq2seq = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q", "v"],
)

# roberta, bert: query, value
# opt: v_proj, q_proj
# llama: v_proj, q_proj
# mistral: v_proj, q_proj
