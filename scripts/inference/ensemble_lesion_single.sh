#!/bin/bash

echo "Started at: $(date)"

# CDIS modality ensembles
echo "Running CDIS ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running CDIS ensemble with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/lesion/cdis/$model/cdis-$model"
done

# DWI modality ensembles
echo "Running DWI ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running DWI ensemble with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/lesion/dwi/$model/dwi-$model"
done

# ADC modality ensembles
echo "Running ADC ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running ADC ensemble with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/lesion/adc/$model/adc-$model"
done

echo "Lesion ensemble benchmarking completed at: $(date)"