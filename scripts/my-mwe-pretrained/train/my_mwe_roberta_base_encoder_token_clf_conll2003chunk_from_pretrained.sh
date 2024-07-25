#!/bin/bash
 
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=16GB
#PBS -o output
#PBS -e output

cd ${PBS_O_WORKDIR:-""}

MODEL_ID="roberta-base"
LR=0.0002 # 2e-4
DO_LOWER_CASE=False
EPOCHS=5
BATCH_SIZE=16
DATASET_NAME="conll2003chunk"
PRETRAINED_MODELS_SAVED_PATH="models/bookcorpus/my-mwe-roberta-encoder-pretrain"
MODEL_SAVE_PATH="models/conll2003chunk/my-mwe-roberta-encoder-from-pretrained"
TASK_TYPE="token_clf"
SEEDS=(120 121 122 123 124)
NAME="my_mwe_roberta_base_encoder_conll2003chunk_from_pretrained"
IS_DECODER=False

export IMAGE_PATH="/lustre/home/ddukic/generative_ie.sif"

/lustre/home/ddukic/scripts/accelerate-singlenode.sh /lustre/home/ddukic/generative-ie/baselines/train_pretrained_mwe_model_token_clf.py \
	--model_id "$MODEL_ID" \
    --lr $LR \
    --do_lower_case $DO_LOWER_CASE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_name "$DATASET_NAME" \
    --pretrained_models_saved_path "$PRETRAINED_MODELS_SAVED_PATH" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --task_type "$TASK_TYPE" \
    --seeds ${SEEDS[@]} \
    --name "$NAME" \
    --is_decoder $IS_DECODER
