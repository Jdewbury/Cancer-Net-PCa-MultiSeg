import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.data_helpers import get_image_and_mask_paths, load_image, normalize_intensity
from pathlib import Path
from sklearn.model_selection import KFold
from typing import List, Dict


class CancerNetPCaDataset(Dataset):
    def __init__(
        self,
        patient_files: Dict[str, dict],
        patient_ids: List[str],
        modalities: List[str] = ["cdis"],
        target_size: tuple = (128, 128),
        lesion_mask: bool = True,
    ):
        self.patient_files = patient_files
        self.patient_ids = patient_ids
        self.modalities = modalities
        self.target_size = target_size
        self.lesion_mask = lesion_mask

        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(target_size),  # uses bilinear interp
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    target_size, interpolation=transforms.InterpolationMode.NEAREST
                ),  # nearest interp for mask
                transforms.ToTensor(),
            ]
        )

        # load and prepare data
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for p_id in self.patient_ids:
            modality_imgs = []
            for m in self.modalities:
                # load and normalize the image
                img_path = self.patient_files[p_id]["images"][m]
                mod_img = load_image(img_path, m)
                mod_img = normalize_intensity(mod_img)

                # add channel dimension if needed
                if len(mod_img.shape) == 3:
                    mod_img = mod_img[None, ...]

                modality_imgs.append(mod_img)

            # stack modalities along channel dimension
            img_np = np.concatenate(modality_imgs, axis=0)

            if self.lesion_mask:
                mask_path = self.patient_files[p_id]["masks"]["lesion"]
            else:
                mask_path = self.patient_files[p_id]["masks"]["prostate"]

            mask_np = np.load(mask_path)

            # align mask with image
            mask_t = np.transpose(mask_np, (2, 1, 0))
            mask = np.flip(mask_t, axis=1)
            mask = (mask > 0).astype(np.float32)

            data.append((img_np, mask))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_volume, mask_volume = self.data[idx]

        c, h, w, d = img_volume.shape

        img_resized = torch.zeros(c, self.target_size[0], self.target_size[1], d)
        mask_resized = torch.zeros(1, self.target_size[0], self.target_size[1], d)

        for d_idx in range(d):
            for c_idx in range(c):
                img_slice = img_volume[c_idx, :, :, d_idx].astype(np.float32)
                img_resized_slice = self.img_transform(img_slice)
                img_resized[c_idx, :, :, d_idx] = img_resized_slice

            mask_slice = mask_volume[:, :, d_idx].astype(np.float32)
            mask_resized_slice = self.mask_transform(mask_slice)
            mask_resized[0, :, :, d_idx] = mask_resized_slice

        return img_resized, mask_resized


class CancerNetPCa:
    def __init__(
        self,
        img_dirs: List[str],
        mask_dir: str,
        modalities: List[str] = ["cdis"],
        num_folds: int = 5,
        fold_idx: int = 0,
        test_split: float = 0.15,
        batch_size: int = 10,
        lesion_mask: bool = True,
        seed: int = 42,
        num_workers: int = None,
        target_size: float = (128, 128),
    ):
        self.img_dirs = [Path(f) for f in img_dirs if not isinstance(f, Path)]
        self.mask_dir = Path(mask_dir) if not isinstance(mask_dir, Path) else mask_dir
        self.modalities = modalities
        self.num_folds = num_folds
        self.fold_idx = fold_idx
        self.test_split = test_split
        self.batch_size = batch_size
        self.lesion_mask = lesion_mask
        self.seed = seed
        self.target_size = target_size

        np.random.seed(seed)

        if num_workers is None:
            self.num_workers = max(1, os.cpu_count() - 2)
        else:
            self.num_workers = num_workers

        self._create_data_folds()
        self._create_dataloader()

    def _create_data_folds(self):
        # get image and mask paths
        patient_files = get_image_and_mask_paths(
            self.img_dirs, self.modalities, self.mask_dir
        )

        print(len(patient_files))

        patient_ids = patient_files.keys()
        patient_ids = np.random.permutation(patient_ids)
        test_size = int(len(patient_ids) * self.test_split)
        test_patients, train_val_patients = np.split(patient_ids, [test_size])

        # create k folds
        folds = {}
        k_folds = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        for fold_idx, (train_idx, val_idx) in enumerate(
            k_folds.split(train_val_patients)
        ):
            train_patients = [train_val_patients[i] for i in train_idx]
            val_patients = [train_val_patients[i] for i in val_idx]

            folds[fold_idx] = {"train": train_patients, "val": val_patients}

        self.train_dataset = CancerNetPCaDataset(
            patient_files=patient_files,
            patient_ids=folds[self.fold_idx]["train"],
            modalities=self.modalities,
            target_size=self.target_size,
            lesion_mask=self.lesion_mask,
        )

        self.val_dataset = CancerNetPCaDataset(
            patient_files=patient_files,
            patient_ids=folds[self.fold_idx]["val"],
            modalities=self.modalities,
            target_size=self.target_size,
            lesion_mask=self.lesion_mask,
        )

        self.test_dataset = CancerNetPCaDataset(
            patient_files=patient_files,
            patient_ids=test_patients,
            modalities=self.modalities,
            target_size=self.target_size,
            lesion_mask=self.lesion_mask,
        )

        print(
            f"Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} validation, {len(self.test_dataset)} test samples"
        )

    def _create_dataloader(self):
        self.train = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
