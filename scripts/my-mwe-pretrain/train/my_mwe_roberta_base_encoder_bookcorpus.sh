#!/bin/bash
 
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=16:mem=64GB
#PBS -o output
#PBS -e output

cd ${PBS_O_WORKDIR:-""}

LR=0.0002 # 2e-4
MODEL_ID="roberta-base"
DO_LOWER_CASE=False
EPOCHS=10
BATCH_SIZE=64
MODEL_SAVE_PATH="models/bookcorpus/my-mwe-roberta-encoder-pretrain"
SEED=120
NAME="my_mwe_roberta_base_encoder_pretrain"
IS_DECODER=False

export IMAGE_PATH="/lustre/home/ddukic/generative_ie.sif"

/lustre/home/ddukic/scripts/accelerate-singlenode.sh /lustre/home/ddukic/generative-ie/baselines/train_my_mwe_model_lm.py \
	--lr $LR \
	--model_id "$MODEL_ID" \
    --do_lower_case $DO_LOWER_CASE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --model_save_path "$MODEL_SAVE_PATH" \
    --seed $SEED \
    --name "$NAME" \
    --is_decoder $IS_DECODER
