#!/bin/bash
 
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=16GB
#PBS -o output
#PBS -e output

cd ${PBS_O_WORKDIR:-""}

MODEL_ID="mistralai/Mistral-7B-v0.1"
DO_LOWER_CASE=False
BATCH_SIZE=16
DATASET_NAME="conll2003chunk"
BASE_MODEL_NAME="mistral"
MODELS_SAVED_PATH="models/conll2003chunk/mistral-7b-instruction-tuning-latest"
SPLIT="test"
NAME="evaluate_mistral_7b_conll2003chunk_test_instruction_tuning_batched_latest"

module load scientific/pytorch/2.0.0-ngc

export IMAGE_PATH="/lustre/home/ddukic/generative_ie_instruction_tuning.sif"

run-singlegpu.sh /lustre/home/ddukic/generative-ie/baselines/eval_decoder_model_instruction_tuning.py \
	--model_id "$MODEL_ID" \
    --do_lower_case $DO_LOWER_CASE \
    --batch_size $BATCH_SIZE \
    --dataset_name "$DATASET_NAME" \
    --base_model_name "$BASE_MODEL_NAME" \
    --models_saved_path "$MODELS_SAVED_PATH" \
    --split "$SPLIT" \
    --name "$NAME"
