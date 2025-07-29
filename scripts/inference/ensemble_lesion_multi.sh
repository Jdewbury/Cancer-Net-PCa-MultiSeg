#!/bin/bash

echo "Started at: $(date)"

# DWI + ADC ensembles
echo "Running DWI+ADC ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running DWI+ADC ensemble with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/lesion/dwi-adc/$model/dwi-adc-$model"
done

# CDIS + DWI ensembles
echo "Running CDIS+DWI ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running CDIS+DWI ensemble with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/lesion/cdis-dwi/$model/cdis-dwi-$model"
done

# CDIS + ADC ensembles
echo "Running CDIS+ADC ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running CDIS+ADC ensemble with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/lesion/cdis-adc/$model/cdis-adc-$model"
done

# CDIS + DWI + ADC ensembles
echo "Running CDIS+DWI+ADC ensemble experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running CDIS+DWI+ADC ensemble with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/ensemble.py \
        --experiment_dir "results/benchmark/lesion/cdis-dwi-adc/$model/cdis-dwi-adc-$model"
done

echo "Multi-modal ensemble benchmarking completed at: $(date)"