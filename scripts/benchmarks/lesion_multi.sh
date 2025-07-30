#!/bin/bash

echo "Started at: $(date)"

# DWI + ADC (without CDIS baseline)
echo "Running DWI+ADC experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running DWI+ADC with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/train.py \
        --experiment_name "dwi-adc-$model" \
        --model $model \
        --modalities dwi adc \
        --img_dirs data_2 data_2 \
        --mask_dir data_2 \
        --output_dir "results/benchmark/lesion/dwi-adc/$model" \
        --epochs 200 --lr_patience 15 --lr_step 0.5 \
        --early_stopping_patience 50 --min_improvement 0.001 \
        --scheduler plateau --optimizer adamw --use_lesion_mask true
done

# CDIS + DWI 
echo "Running CDIS+DWI experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running CDIS+DWI with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/train.py \
        --experiment_name "cdis-dwi-$model" \
        --model $model \
        --modalities cdis dwi \
        --img_dirs data/images data_2 \
        --mask_dir data_2 \
        --output_dir "results/benchmark/lesion/cdis-dwi/$model" \
        --epochs 200 --lr_patience 15 --lr_step 0.5 \
        --early_stopping_patience 50 --min_improvement 0.001 \
        --scheduler plateau --optimizer adamw --use_lesion_mask true
done

# CDIS + ADC
echo "Running CDIS+ADC experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running CDIS+ADC with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/train.py \
        --experiment_name "cdis-adc-$model" \
        --model $model \
        --modalities cdis adc \
        --img_dirs data/images data_2 \
        --mask_dir data_2 \
        --output_dir "results/benchmark/lesion/cdis-adc/$model" \
        --epochs 200 --lr_patience 15 --lr_step 0.5 \
        --early_stopping_patience 50 --min_improvement 0.001 \
        --scheduler plateau --optimizer adamw --use_lesion_mask true
done

# CDIS + DWI + ADC (all three modalities)
echo "Running CDIS+DWI+ADC experiments..."
for model in swinunetr attentionunet segresnet unet; do 
    echo "  Running CDIS+DWI+ADC with $model..."
    CUDA_VISIBLE_DEVICES=0 poetry run python scripts/train.py \
        --experiment_name "cdis-dwi-adc-$model" \
        --model $model \
        --modalities cdis dwi adc \
        --img_dirs data/images data_2 data_2 \
        --mask_dir data_2 \
        --output_dir "results/benchmark/lesion/cdis-dwi-adc/$model" \
        --epochs 200 --lr_patience 15 --lr_step 0.5 \
        --early_stopping_patience 50 --min_improvement 0.001 \
        --scheduler plateau --optimizer adamw --use_lesion_mask true
done

echo "Multi-modal lesion mask benchmarking completed at: $(date)"