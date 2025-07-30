#!/bin/bash

echo "Started at: $(date)"

# CDIS modality ensembles
echo "Running CDIS ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running CDIS ensemble with $model..."
    CUDA_VISIBLE_DEVICES=1 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/prostate/cdis/$model/cdis-$model"
done

# DWI modality ensembles
echo "Running DWI ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running DWI ensemble with $model..."
    CUDA_VISIBLE_DEVICES=1 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/prostate/dwi/$model/dwi-$model"
done

# ADC modality ensembles
echo "Running ADC ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running ADC ensemble with $model..."
    CUDA_VISIBLE_DEVICES=1 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/prostate/adc/$model/adc-$model"
done

echo "Prostate ensemble benchmarking completed at: $(date)"