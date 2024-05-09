#!/bin/bash
MODEL_NAME=lemma-llama
PEFT_PATH=MLP-Lemma/lemma-llama-pt9k-sft2.0k

# REAL SCRIPTS
CUDA_VISIBLE_DEVICES=0,1,2,3 python pred.py \
    --model $MODEL_NAME \
    --e \
    --peft_path $PEFT_PATH

wait

CUDA_VISIBLE_DEVICES=0,1,2,3 python pred.py \
    --model $MODEL_NAME \
    --peft_path $PEFT_PATH

wait

python eval.py --model $MODEL_NAME

wait

python eval.py --model $MODEL_NAME --e
