#!/bin/bash

echo "Started at: $(date)"

# CDIS modality
echo "Running CDIS experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running CDIS with $model..."
    CUDA_VISIBLE_DEVICES=1 poetry run python scripts/train.py \
        --experiment_name "cdis-$model" \
        --model $model \
        --modalities cdis \
        --img_dirs data/images \
        --mask_dir data_2 \
        --output_dir "benchmark/prostate/cdis/$model" \
        --epochs 200 --lr_patience 15 --lr_step 0.5 \
        --early_stopping_patience 50 --min_improvement 0.001 \
        --scheduler plateau --optimizer adamw --use_lesion_mask false
done

# DWI modality
echo "Running DWI experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running DWI with $model..."
    CUDA_VISIBLE_DEVICES=1 poetry run python scripts/train.py \
        --experiment_name "dwi-$model" \
        --model $model \
        --modalities dwi \
        --img_dirs data_2 \
        --mask_dir data_2 \
        --output_dir "benchmark/prostate/dwi/$model" \
        --epochs 200 --lr_patience 15 --lr_step 0.5 \
        --early_stopping_patience 50 --min_improvement 0.001 \
        --scheduler plateau --optimizer adamw --use_lesion_mask false
done

# ADC modality
echo "Running ADC experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running ADC with $model..."
    CUDA_VISIBLE_DEVICES=1 poetry run python scripts/train.py \
        --experiment_name "adc-$model" \
        --model $model \
        --modalities adc \
        --img_dirs data_2 \
        --mask_dir data_2 \
        --output_dir "benchmark/prostate/adc/$model" \
        --epochs 200 --lr_patience 15 --lr_step 0.5 \
        --early_stopping_patience 50 --min_improvement 0.001 \
        --scheduler plateau --optimizer adamw --use_lesion_mask false
done

echo "Prostate mask benchmarking completed at: $(date)"