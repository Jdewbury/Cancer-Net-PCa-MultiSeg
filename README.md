# Cancer-Net PCa-Seg: Benchmarking Deep Learning Models for Prostate Cancer Segmentation Using Synthetic Correlated Diffusion Imaging

![Sample dataset](https://github.com/Jdewbury/Cancer-Net-PCa-Seg/blob/main/src/sample.png)

This repository contains modules and instructions for replicating and extending experiments featured in our paper:
- Training scripts [scripts/benchmarks](scripts/benchmarks) to benchmark the select architectures on the PCa CDI<sup>s</sup> data and other available modalities
- Ensemble inference scripts [scripts/inference](scripts/inference) for combining predictions across k-fold models using weighted soft ensembling


## Dataset
Cancer-Net PCa-Data is an open access benchmark dataset of volumetric correlated diffusion imaging (CDIs) data acquisitions of prostate cancer patients. Cancer-Net PCa-Data is a part of the Cancer-Net open source initiative dedicated to advancement in machine learning and imaging research to aid clinicians in the global fight against cancer.

The volumetric CDIs data acquisitions in the Cancer-Net PCa-Data dataset were generated from a patient cohort of 200 patient cases acquired at Radboud University Medical Centre (Radboudumc) in the Prostate MRI Reference Center in Nijmegen, The Netherlands and made available as part of the SPIE-AAPM-NCI PROSTATEx Challenges. Masks derived from the PROSTATEx_masks repository are also provided which label regions of healthy prostate tissue, clinically significant prostate cancer (csPCa), and clinically insignificant prostate cancer (insPCa).

This [dataset](https://www.kaggle.com/datasets/hgunraj/cancer-net-pca-data) is being used to train and validate our Cancer-Net PCa-Seg models for PCa lesion segmentation from CDIs data acquisitions.

## Model Training and Inference
### Single Model Training
If you want to train and evaluate a network from scratch, you can run the provided training scripts:

<b>Lesion Segmentation</b>
```
# single modality experiments
./scripts/benchmarks/lesion_single.sh

# multi-model experiments
./scripts/benchmarks/lesion_multi.sh
```
<b>Prostate Segmentation</b>
```
# single modality experiments
./scripts/benchmarks/prostate_single.sh

# multi-model experiments
./scripts/benchmarks/prostate_multi.sh
```

### Ensemble Inference
After training is finished, you can run ensemble inference to combine predictions from all k-fold models using weighted soft ensembling:

<b>Lesion Segmentation Ensembling</b>
```
# single modality ensembles
./scripts/inference/ensemble_lesion_single.sh

# multi-model ensembles
./scripts/inference/ensemble_lesion_multi.sh
```
<b>Prostate Segmentation Ensembling</b>
```
# single modality ensembles
./scripts/inference/ensemble_prostate_single.sh

# multi-model ensembles
./scripts/inference/ensemble_prostate_multi.sh
```

### Individual Experiment Training
You can also train your own individual experiments using the main training script:
```
poetry run python scripts/train.py \
    --experiment_name "cdis-unet" \
    --model unet \
    --modalities cdis \
    --img_dirs data/images \
    --mask_dir data_2 \
    --output_dir "results/benchmark/lesion/cdis/unet" \
    --epochs 200 \
    --use_lesion_mask true
```
### Individual Ensemble Inference
You can run ensemble inference on specific experiments:
```
poetry run python scripts/ensemble.py \
    --experiment_dir "results/benchmark/lesion/cdis/unet/cdis-unet"
```

## Configuration
You can modify the default arguments in the [config](utils/config.py) file, or specify CLI arguments when running individual scripts.

Full CLI:
```
usage: [train.py/ensemble.py] [-h] [--img_dirs IMG_DIRS [IMG_DIRS ...]] [--mask_dir MASK_DIR] [--modalities {cdis,dwi,adc} [{cdis,dwi,adc} ...]] 
                [--target_size_h TARGET_SIZE_H] [--target_size_w TARGET_SIZE_W] [--use_lesion_mask {true,false}] [--batch_size BATCH_SIZE] [--epochs EPOCHS]  
                [--learning_rate LEARNING_RATE] [--num_folds NUM_FOLDS] [--test_split TEST_SPLIT] [--seed SEED] [--early_stopping_patience EARLY_STOPPING_PATIENCE] 
                [--min_improvement MIN_IMPROVEMENT] [--model {segresnet,unet,swinunetr,attentionunet,mambaunet}] [--init_filters INIT_FILTERS]
                [--optimizer {adam,adamw,sgd}] [--scheduler {None,step,cosine,plateau}] [--lr_step LR_STEP] [--lr_patience LR_PATIENCE] [--output_dir OUTPUT_DIR] 
                [--experiment_name EXPERIMENT_NAME] [--experiment_dir EXPERIMENT_DIR]

CancerNet-PCa Segmentation

options:
  -h, --help            show this help message and exit
  --img_dirs IMG_DIRS [IMG_DIRS ...]
                        Directory containing image data. Pass multiple directories for more than one modality. Default: ['data/images']
  --mask_dir MASK_DIR   Directory containing mask data. Default: data_2
  --modalities {cdis,dwi,adc} [{cdis,dwi,adc} ...]
                        One or more image modalities to evaluate. Default: ['cdis']
  --target_size_h TARGET_SIZE_H
                        Target height of input image. Default: 128
  --target_size_w TARGET_SIZE_W
                        Target width of input image. Default: 128
  --use_lesion_mask {true,false}
                        Whether to use lesion mask (true) or prostate mask (false). Default: True
  --batch_size BATCH_SIZE
                        Batch size for the training and validation loops. Default: 1
  --epochs EPOCHS       Total number of training epochs. Default: 200
  --learning_rate LEARNING_RATE
                        Initial learning rate for training. Default: 0.001
  --num_folds NUM_FOLDS
                        Number of K folds to evaluate over. Default: 5
  --test_split TEST_SPLIT
                        Percent allocation of dataset to the test set. Default: 0.1
  --seed SEED           Seed to use for splitting dataset. Default: 42
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Number of epochs without improvement for early stopping. Default: 50
  --min_improvement MIN_IMPROVEMENT
                        Minimum dice improvement to reset patience counter. Default: 0.001
  --model {segresnet,unet,swinunetr,attentionunet,mambaunet}
                        Model architecture to be used for training. Default: unet
  --init_filters INIT_FILTERS
                        Number of filters for model. Default: 32
  --optimizer {adam,adamw,sgd}
                        Optimizer to use for training. Default: adamw
  --scheduler {None,step,cosine,plateau}
                        Learning rate scheduler to use. Default: plateau
  --lr_step LR_STEP     Learning rate step size. Default: 0.5
  --lr_patience LR_PATIENCE
                        Learning rate patience before reduction. Default: 15
  --output_dir OUTPUT_DIR
                        Output directory to save training results. Default: results
  --experiment_name EXPERIMENT_NAME
                        Name of experiment. Default: cancer-net-pca-seg
  --experiment_dir EXPERIMENT_DIR
                        Path to experiment directory containing fold results. Default: results/cancer-net-pca-seg
```
