#!/bin/bash
 
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=16GB
#PBS -o output
#PBS -e output

cd ${PBS_O_WORKDIR:-""}

MODEL_ID="roberta-base"
DO_LOWER_CASE=False
BATCH_SIZE=256
DATASET_NAME="absa-restaurants"
MODELS_SAVED_PATH="models/absa-restaurants/my-mwe-roberta-decoder-from-pretrained"
SPLIT="test"
TASK_TYPE="token_clf"
NAME="evaluate_my_mwe_roberta_base_decoder_absa-restaurants_from_pretrained_test"

module load scientific/pytorch/2.0.0-ngc

export IMAGE_PATH="/lustre/home/ddukic/generative_ie.sif"

run-singlegpu.sh /lustre/home/ddukic/generative-ie/baselines/eval_pretrained_mwe_model_token_clf.py \
    --model_id "$MODEL_ID" \
    --do_lower_case $DO_LOWER_CASE \
    --batch_size $BATCH_SIZE \
    --dataset_name "$DATASET_NAME" \
    --models_saved_path "$MODELS_SAVED_PATH" \
    --split "$SPLIT" \
    --task_type "$TASK_TYPE" \
    --name "$NAME"
