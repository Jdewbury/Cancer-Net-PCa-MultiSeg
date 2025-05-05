# Cancer-Net PCa-Seg: Benchmarking Deep Learning Models for Prostate Cancer Segmentation Using Synthetic Correlated Diffusion Imaging

![Sample dataset](https://github.com/Jdewbury/Cancer-Net-PCa-Seg/blob/main/src/sample.png)
<br><br>
Prostate cancer (PCa) is the most prevalent cancer among men in the United States, accounting for nearly 300,000 cases, 29\% of all diagnoses and 35,000 total deaths in 2024. Traditional screening methods such as prostate-specific antigen (PSA) testing and magnetic resonance imaging (MRI) have been pivotal in diagnosis, but have faced limitations in specificity and generalizability. In this paper, we explore the potential of enhancing PCa lesion segmentation using a novel MRI modality called synthetic correlated diffusion imaging (CDI<sup>s</sup>). We employ several state-of-the-art deep learning models, including U-Net, SegResNet, Swin UNETR, Attention U-Net, and LightM-UNet, to segment PCa lesions from a 200 CDI<sup>s</sup> patient cohort. We find that SegResNet achieved superior segmentation performance with a Dice-Sørensen coefficient (DSC) of $76.68 \pm 0.8$. Notably, the Attention U-Net, while slightly less accurate (DSC $74.82 \pm 2.0$), offered a favorable balance between accuracy and computational efficiency. Our findings demonstrate the potential of deep learning models in improving PCa lesion segmentation using CDI<sup>s</sup> to enhance PCa management and clinical support. 
<br><br>
This repository contains modules and instructions for replicating and extending experiments featured in our paper:
- A training and inference script [CancerNetPCaSeg.py](CancerNetPCaSeg.py) to train and evaluate the select architectures on the PCa CDI<sup>s</sup> data

## Dataset
Cancer-Net PCa-Data is an open access benchmark dataset of volumetric correlated diffusion imaging (CDIs) data acquisitions of prostate cancer patients. Cancer-Net PCa-Data is a part of the Cancer-Net open source initiative dedicated to advancement in machine learning and imaging research to aid clinicians in the global fight against cancer.

The volumetric CDIs data acquisitions in the Cancer-Net PCa-Data dataset were generated from a patient cohort of 200 patient cases acquired at Radboud University Medical Centre (Radboudumc) in the Prostate MRI Reference Center in Nijmegen, The Netherlands and made available as part of the SPIE-AAPM-NCI PROSTATEx Challenges. Masks derived from the PROSTATEx_masks repository are also provided which label regions of healthy prostate tissue, clinically significant prostate cancer (csPCa), and clinically insignificant prostate cancer (insPCa).

This [dataset](https://www.kaggle.com/datasets/hgunraj/cancer-net-pca-data) is being used to train and validate our Cancer-Net PCa-Seg models for PCa lesion segmentation from CDIs data acquisitions.

## Model Training and Inference
If you want to train and evaluate a network from scratch, you can run the provided file using the default arguments:
```
python3 CancerNetPCaSeg.py
```
You can modify the default arguments in the [config](utils/config.py) file, or specify CLI arguments when running

Full CLI:
```
usage: CancerNetPCaSeg.py [-h] [--img_dirs IMG_DIRS [IMG_DIRS ...]] [--mask_dir MASK_DIR] [--modalities {cdis,dwi,adc} [{cdis,dwi,adc} ...]]
                            [--target_size TARGET_SIZE] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--num_folds NUM_FOLDS]
                            [--test_split TEST_SPLIT] [--seed SEED] [--early_stopping_patience EARLY_STOPPING_PATIENCE]
                            [--model {segresnet,unet,swinunetr,attentionunet,mambaunet}] [--init_filters INIT_FILTERS] [--optimizer {adam,adamw,sgd}]
                            [--scheduler {None,step,cosine}] [--lr_step LR_STEP] [--output_dir OUTPUT_DIR] [--experiment_name EXPERIMENT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --img_dirs IMG_DIRS [IMG_DIRS ...]
                        Directory containing image data. Pass multiple directories for more than one modality. Default: ['data/images']
  --mask_dir MASK_DIR   Directory containing mask data. Default: data_2
  --modalities {cdis,dwi,adc} [{cdis,dwi,adc} ...]
                        One or more image modalities to evaluate. Default: ['cdis']
  --target_size TARGET_SIZE
                        Target size of input image into model. Default: (128, 128)
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
                        Number of epochs without improvement for early stopping. Default: 15
  --model {segresnet,unet,swinunetr,attentionunet,mambaunet}
                        Model architecture to be used for training. Default: unet
  --init_filters INIT_FILTERS
                        Number of filters for model. Default: 32
  --optimizer {adam,adamw,sgd}
                        Optimizer to use for training. Default: adam
  --scheduler {None,step,cosine}
                        Learning rate scheduler to use. Default: step
  --lr_step LR_STEP     Learning rate step size. Default: 0.1
  --output_dir OUTPUT_DIR
                        Output directory to save training results. Default: results
  --experiment_name EXPERIMENT_NAME
                        Name of experiment. Default: cancer-net-pca-seg
```
