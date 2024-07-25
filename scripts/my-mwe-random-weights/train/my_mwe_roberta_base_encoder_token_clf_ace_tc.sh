#!/bin/bash
 
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=16GB
#PBS -o output
#PBS -e output

cd ${PBS_O_WORKDIR:-""}

LR=0.0002 # 2e-4
MODEL_ID="roberta-base"
DO_LOWER_CASE=False
EPOCHS=5
BATCH_SIZE=16
DATASET_NAME="ace-tc"
MODEL_SAVE_PATH="models/ace-tc/my-mwe-roberta-encoder-random-init"
TASK_TYPE="token_clf"
SEEDS=(120 121 122 123 124)
NAME="my_mwe_roberta_base_encoder_ace_tc"
IS_DECODER=False

export IMAGE_PATH="/lustre/home/ddukic/generative_ie.sif"

/lustre/home/ddukic/scripts/accelerate-singlenode.sh /lustre/home/ddukic/generative-ie/baselines/train_my_mwe_model_token_clf.py \
	--lr $LR \
	--model_id "$MODEL_ID" \
    --do_lower_case $DO_LOWER_CASE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_name "$DATASET_NAME" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --task_type "$TASK_TYPE" \
    --seeds ${SEEDS[@]} \
    --name "$NAME" \
    --is_decoder $IS_DECODER
