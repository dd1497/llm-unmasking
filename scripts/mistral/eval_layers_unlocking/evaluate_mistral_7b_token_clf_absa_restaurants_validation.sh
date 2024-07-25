#!/bin/bash
 
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=16GB
#PBS -o output
#PBS -e output

cd ${PBS_O_WORKDIR:-""}

MODEL_ID="mistralai/Mistral-7B-v0.1"
DO_LOWER_CASE=False
BATCH_SIZE=16
DATASET_NAME="absa-restaurants"
BASE_MODEL_NAME="mistral"
MODELS_SAVED_PATH="models/absa-restaurants/mistral-7b-layers-unlocking"
SPLIT="validation"
TASK_TYPE="token_clf"
NAME="evaluate_mistral_7b_absa-restaurants_validation_layers_unlocking_new"

module load scientific/pytorch/2.0.0-ngc

export IMAGE_PATH="/lustre/home/ddukic/generative_ie.sif"

run-singlegpu.sh /lustre/home/ddukic/generative-ie/baselines/eval_decoder_model_layers_unlocking.py \
	--model_id "$MODEL_ID" \
    --do_lower_case $DO_LOWER_CASE \
    --batch_size $BATCH_SIZE \
    --dataset_name "$DATASET_NAME" \
    --base_model_name "$BASE_MODEL_NAME" \
    --models_saved_path "$MODELS_SAVED_PATH" \
    --split "$SPLIT" \
    --task_type "$TASK_TYPE" \
    --name "$NAME"
