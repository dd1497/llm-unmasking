#!/bin/bash
 
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=16GB
#PBS -o output
#PBS -e output

cd ${PBS_O_WORKDIR:-""}

LR=0.0002 # 2e-4
MODEL_ID="mistralai/Mistral-7B-v0.1"
DO_LOWER_CASE=False
EPOCHS=5
BATCH_SIZE=16
DATASET_NAME="conll2003chunk"
BASE_MODEL_NAME="mistral"
MODEL_SAVE_PATH="models/conll2003chunk/mistral-7b-layers-unlocking-from-tapt-clm"
TASK_TYPE="token_clf"
SEEDS=(120 121 122 123 124)
NAME="mistral_7b_conll2003chunk_layers_unlocking_tapt_clm"
TAPT="clm"

export IMAGE_PATH="/lustre/home/ddukic/generative_ie.sif"

/lustre/home/ddukic/scripts/accelerate-singlenode.sh /lustre/home/ddukic/generative-ie/baselines/train_decoder_model_layers_unlocking.py \
	--lr $LR \
	--model_id "$MODEL_ID" \
    --do_lower_case $DO_LOWER_CASE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_name "$DATASET_NAME" \
    --base_model_name "$BASE_MODEL_NAME" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --task_type "$TASK_TYPE" \
    --seeds ${SEEDS[@]} \
    --name "$NAME" \
    --tapt "$TAPT"
