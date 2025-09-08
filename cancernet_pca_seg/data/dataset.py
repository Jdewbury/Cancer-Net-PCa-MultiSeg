import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset


def get_bounding_box(mask: np.ndarray, padding: int = 10) -> Tuple[int]:
    """Get the bounding box of the input mask (y, x, z)

    Args:
        mask: input mask to get bounding box
        padding: desired padding to apply to mask

    Returns:
        Tuple with bounding box coordinates (y, x, z)
    """
    y_max, x_max, z_max = mask.shape
    nonzero_idxs = np.nonzero(mask)

    y_min, y_max = max(0, nonzero_idxs[0].min() - padding), min(
        y_max, nonzero_idxs[0].max() + padding + 1
    )
    x_min, x_max = max(0, nonzero_idxs[1].min() - padding), min(
        x_max, nonzero_idxs[1].max() + padding + 1
    )
    z_min, z_max = max(0, nonzero_idxs[2].min() - padding), min(
        z_max, nonzero_idxs[2].max() + padding + 1
    )

    if y_min >= y_max or x_min >= x_max or z_min >= z_max:
        raise ValueError(
            f"Invalid bounding box: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}], z=[{z_min}:{z_max}]"
        )

    return (y_min, y_max, x_min, x_max, z_min, z_max)


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

    def _load_image(self, file_path: Path, modality: str = "cdis") -> np.ndarray:
        if modality == "cdis":
            img_sitk = sitk.ReadImage(file_path)
            img = sitk.GetArrayFromImage(img_sitk).astype(np.uint8)
            img = np.nan_to_num(img).astype(np.float32)
            return np.transpose(img, (2, 1, 0))

        elif modality == "adc":
            img_np = np.load(file_path, allow_pickle=True)
            img_t = np.transpose(img_np, (2, 1, 0))
            return np.flip(img_t, axis=1)

        elif modality == "dwi":
            img_np = np.load(file_path, allow_pickle=True)
            img_t = np.transpose(img_np, (0, 3, 2, 1))
            return np.flip(img_t, axis=2)
        else:
            raise ValueError(
                f"Invalid modality name: {modality}. Choose from 'cdis', 'dwi', or 'adc'."
            )

    def _normalize_intensity(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 4:
            norm_img = np.zeros_like(img)
            for c in range(img.shape[0]):
                channel = img[c]
                min_val, max_val = channel.min(), channel.max()
                if max_val > min_val:
                    norm_img[c] = (channel - min_val) / (max_val - min_val)
                else:
                    norm_img[c] = channel
            return norm_img
        else:
            min_val, max_val = img.min(), img.max()
            if max_val > min_val:
                return (img - min_val) / (max_val - min_val)
            else:
                return img

    def _prepare_data(self):
        data = []
        for p_id in self.patient_ids:
            modality_imgs = []
            for m in self.modalities:
                # load and normalize the image
                img_path = self.patient_files[p_id]["images"][m]
                mod_img = self._load_image(img_path, m)
                norm_img = self._normalize_intensity(mod_img)

                # add channel dimension if needed
                if len(norm_img.shape) == 3:
                    norm_img = norm_img[None, ...]

                modality_imgs.append(norm_img)

            # stack modalities along channel dimension
            img_np = np.concatenate(modality_imgs, axis=0)

            mask_type = "lesion" if self.lesion_mask else "prostate"
            mask_path = self.patient_files[p_id]["masks"][mask_type]
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
        target_size: tuple = (128, 128),
        lesion_mask: bool = True,
        num_folds: int = 5,
        fold_idx: int = 0,
        test_split: float = 0.15,
        batch_size: int = 10,
        seed: int = 42,
        num_workers: int = None,
    ):
        self.img_dirs = [Path(f) if not isinstance(f, Path) else f for f in img_dirs]
        self.mask_dir = Path(mask_dir) if not isinstance(mask_dir, Path) else mask_dir
        self.modalities = modalities
        self.target_size = target_size
        self.lesion_mask = lesion_mask
        self.num_folds = num_folds
        self.fold_idx = fold_idx
        self.test_split = test_split
        self.batch_size = batch_size
        self.seed = seed

        np.random.seed(seed)

        if num_workers is None:
            self.num_workers = max(1, os.cpu_count() - 2)
        else:
            self.num_workers = num_workers

        self.patients = {}
        self._create_data_folds()
        self._create_dataloader()

    def _add_patient_file(
        self, patient_id: str, category: str, file_type: str, file_path: Path
    ) -> None:
        if patient_id not in self.patients:
            self.patients[patient_id] = {"images": {}, "masks": {}}
        self.patients[patient_id][category][file_type] = file_path

    def _find_patient_files(self) -> None:
        for img_dir, modality in zip(self.img_dirs, self.modalities):
            if modality == "cdis":
                pattern = "*.nii"
                file_paths = sorted(img_dir.rglob(pattern))
                for f in file_paths:
                    patient_id = f.name.split("_")[0]
                    self._add_patient_file(
                        patient_id,
                        "images",
                        modality,
                        f,
                    )

            elif modality in ["adc", "dwi"]:
                pattern = f"*{modality.upper()}.npy"
                file_paths = sorted(img_dir.rglob(pattern))
                for f in file_paths:
                    patient_id = f.parent.name
                    self._add_patient_file(
                        patient_id,
                        "images",
                        modality,
                        f,
                    )

            else:
                raise ValueError(f"Unknown modality: {modality}")

        lesion_files = sorted(self.mask_dir.rglob("*lesion_mask.npy"))
        prostate_files = sorted(self.mask_dir.rglob("*prostate_mask.npy"))

        for f in lesion_files:
            patient_id = f.parent.name
            self._add_patient_file(patient_id, "masks", "lesion", f)
        for f in prostate_files:
            patient_id = f.parent.name
            self._add_patient_file(patient_id, "masks", "prostate", f)

    def _create_data_folds(self):
        # get image and mask paths
        self._find_patient_files()
        print(f"Found {len(self.patients)} patients")

        patient_ids = list(self.patients.keys())
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
            patient_files=self.patients,
            patient_ids=folds[self.fold_idx]["train"],
            modalities=self.modalities,
            target_size=self.target_size,
            lesion_mask=self.lesion_mask,
        )

        self.val_dataset = CancerNetPCaDataset(
            patient_files=self.patients,
            patient_ids=folds[self.fold_idx]["val"],
            modalities=self.modalities,
            target_size=self.target_size,
            lesion_mask=self.lesion_mask,
        )

        self.test_dataset = CancerNetPCaDataset(
            patient_files=self.patients,
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
