import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_helpers import list_image_paths, load_image, normalize_intensity
from pathlib import Path

LESION_MASK_NAME = "lesion_mask.npy"
PROSTATE_MASK_NAME = "prostate_mask.npy"


class CancerNetPCaDataset(Dataset):
    def __init__(
        self,
        img_paths: list[Path],
        mask_paths: list[Path],
        modalities: list[str],
        prostate: bool = False,
        transform=None,
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.modalities = modalities
        self.prostate = prostate
        self.transform = transform

        # load and prepare data
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for i in range(len(self.mask_paths)):
            modality_imgs = []
            for m_idx, modality in enumerate(self.modalities):
                # load and normalize the image
                img_path = self.img_paths[m_idx][i]
                mod_img = load_image(img_path, modality)
                mod_img = normalize_intensity(mod_img)

                # add channel dimension if needed
                if len(mod_img.shape) == 3:
                    mod_img = mod_img[None, ...]

                modality_imgs.append(mod_img)

            # stack modalities along channel dimension
            img_np = np.concatenate(modality_imgs, axis=0)

            lesion_path, prostate_path = self.mask_paths[i]
            lesion_np = np.load(lesion_path)
            prostate_np = np.load(prostate_path)

            if self.prostate:
                lesion_np *= prostate_np

            # align mask with image
            mask_t = np.transpose(lesion_np, (2, 1, 0))
            mask = np.flip(mask_t, axis=1)
            mask = (mask > 0).astype(np.float32)

            # extract 2D slices from 3D volumes
            num_slices = min(img_np.shape[2], mask.shape[2])
            for s_idx in range(num_slices):
                if len(img_np.shape) == 4:
                    # take all channels for the current slice
                    img_slice = img_np[:, :, :, s_idx].astype(np.float32)
                else:
                    img_slice = img_np[:, :, s_idx].astype(np.float32)

                mask_slice = mask[:, :, s_idx].astype(np.float32)

                data.append((img_slice, mask_slice))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask = self.data[idx]

        if self.transform is not None:
            if len(img.shape) == 3:
                transformed_channels = []
                for c in range(img.shape[0]):
                    transformed_channels.append(self.transform(img[c]))
                img_slice = torch.cat(transformed_channels, dim=0)
            else:
                img_slice = self.transform(img)

            mask_slice = self.transform(mask)

        return img_slice, mask_slice


class CancerNetPCa:
    def __init__(
        self,
        img_dirs: list[Path],
        mask_dir: Path,
        modalities: list[str] = ["cdis"],
        seed: int = 42,
        batch_size: int = 10,
        train_split: float = 0.7,
        test_split: float = 0.15,
        prostate: bool = False,
        transform=None,
        num_workers: int = None,
    ):
        self.img_dirs = img_dirs
        self.mask_dir = mask_dir
        self.modalities = modalities
        self.batch_size = batch_size
        self.train_split = train_split
        self.test_split = test_split
        self.prostate = prostate
        self.transform = transform

        np.random.seed(seed)

        if num_workers is None:
            self.num_workers = max(1, os.cpu_count() - 2)
        else:
            self.num_workers = num_workers

        self._create_dataset()
        self._create_dataloader()

    def _create_dataset(self):
        # get image and mask paths
        img_paths = np.array(
            [list_image_paths(dir, m) for dir, m in zip(self.img_dirs, self.modalities)]
        )
        lesion_paths = sorted(self.mask_dir.rglob(f"*{LESION_MASK_NAME}"))
        prostate_paths = sorted(self.mask_dir.rglob(f"*{PROSTATE_MASK_NAME}"))

        mask_pairs = np.array([lesion_paths, prostate_paths])

        dataset_size = len(mask_pairs)
        idxs = np.arange(dataset_size)
        np.random.shuffle(idxs)

        train_size = int(self.train_split * dataset_size)
        test_size = int(self.test_split * dataset_size)
        val_size = dataset_size - train_size - test_size

        train_idx = idxs[:train_size]
        test_idx = idxs[train_size : train_size + test_size]
        val_idx = idxs[train_size + test_size :]

        self.train_dataset = CancerNetPCaDataset(
            img_paths=img_paths[:, train_idx],
            mask_paths=mask_pairs[:, train_idx],
            modalities=self.modalities,
            prostate=self.prostate,
            transform=self.transform,
        )

        self.val_dataset = CancerNetPCaDataset(
            img_paths=img_paths[:, val_idx],
            mask_paths=mask_pairs[:, val_idx],
            modalities=self.modalities,
            prostate=self.prostate,
            transform=self.transform,
        )

        self.test_dataset = CancerNetPCaDataset(
            img_paths=img_paths[:, test_idx],
            mask_paths=mask_pairs[:, test_idx],
            modalities=self.modalities,
            prostate=self.prostate,
            transform=self.transform,
        )

        print(
            f"Dataset split: {train_size} train, {val_size} validation, {test_size} test samples"
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
