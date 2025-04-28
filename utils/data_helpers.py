import numpy as np
from pathlib import Path
import SimpleITK as sitk
from typing import List, Dict, Union

LESION_MASK_NAME = "lesion_mask.npy"
PROSTATE_MASK_NAME = "prostate_mask.npy"


class PatientData:
    def __init__(self):
        self.patients = {}

    def _add_image_path(self, patient_id, modality, file_path):
        if patient_id not in self.patients:
            self.patients[patient_id] = {"images": {}, "masks": {}}
        self.patients[patient_id]["images"][modality] = file_path

    def _add_mask_path(self, patient_id, mask_type, file_path):
        if patient_id not in self.patients:
            self.patients[patient_id] = {"images": {}, "masks": {}}
        self.patients[patient_id]["masks"][mask_type] = file_path

    def __getitem__(self, key):
        return self.patients[key]

    def __len__(self):
        return len(self.patients)

    def items(self):
        return self.patients.items()

    def keys(self):
        return np.array(list(self.patients.keys()))

    def values(self):
        return self.patients.values()


def list_image_paths(root_dir: Path, modality: str = "cdis") -> List[Path]:
    """Gather all image paths in directory for specified modality.

    Args:
        root_dir: directory to search for image files
        modality: desired modality of interest

    Returns:
        Sorted list of image file paths
    """
    if modality in ["adc", "dwi"]:
        pattern = f"*{modality.upper()}.npy"
    elif modality == "cdis":
        pattern = "*.nii"
    else:
        raise ValueError(
            f"Invalid modality name: {modality}. Choose from 'cdis', 'dwi', or 'adc'."
        )

    return sorted(root_dir.rglob(pattern))


def get_image_and_mask_paths(
    img_dirs: List[Path], modalities: List[str], mask_dir: Path
) -> Dict[str, dict]:
    """Get the filepaths to the image(s) and masks for each patient.

    Args:
        img_dirs: directories to search for image files
        modalities: corresponding image modalities of image directories
        mask_dir: directory to corresponding segmentation masks

    Returns:
        dictionary containing image and mask paths for each patient
    """
    patient_files = PatientData()

    for dir, m in zip(img_dirs, modalities):
        file_paths = list_image_paths(dir, m)
        if m in ["adc", "dwi"]:
            for f in file_paths:
                patient_files._add_image_path(f.parent.name, m, f)
        elif m == "cdis":
            for f in file_paths:
                patient_files._add_image_path(f.name.split("_")[0], m, f)

    lesion_paths = sorted(mask_dir.rglob(f"*{LESION_MASK_NAME}"))
    prostate_paths = sorted(mask_dir.rglob(f"*{PROSTATE_MASK_NAME}"))

    for f in lesion_paths:
        patient_files._add_mask_path(f.parent.name, "lesion", f)
    for f in prostate_paths:
        patient_files._add_mask_path(f.parent.name, "prostate", f)

    return patient_files


def load_image(file_path: Path, modality: str = "cdis") -> np.ndarray:
    """Load in medical image and convert to numpy array.

    Args:
        file_path: path to image file
        modality: desired modality of interest

    Returns:
        Numpy array object of loaded image
    """
    if modality == "cdis":
        img_sitk = sitk.ReadImage(file_path)
        img = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        img = np.nan_to_num(img)
        img = np.transpose(img, (2, 1, 0))
    elif modality == "adc":
        img_np = np.load(file_path, allow_pickle=True)
        img_t = np.transpose(img_np, (2, 1, 0))
        img = np.flip(img_t, axis=1)
    elif modality == "dwi":
        img_np = np.load(file_path, allow_pickle=True)
        img_t = np.transpose(img_np, (0, 3, 2, 1))
        img = np.flip(img_t, axis=2)
    else:
        raise ValueError(
            f"Invalid modality name: {modality}. Choose from 'cdis', 'dwi', or 'adc'."
        )

    return img


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Normalize intensity of image to range [0, 1].

    Args:
        img: input image array

    Returns:
        Normalized image array
    """
    if len(img.shape) == 4:
        norm_img = np.zeros_like(img)
        for c in range(img.shape[0]):
            img_linear_window = [img[c].min(), img[c].max()]
            img_clip = np.clip(img[c], *img_linear_window)
            norm_img[c] = (img_clip - img_linear_window[0]) / (
                img_linear_window[1] - img_linear_window[0]
            )
    else:
        img_linear_window = [img.min(), img.max()]
        img_clip = np.clip(img, *img_linear_window)
        norm_img = (img_clip - img_linear_window[0]) / (
            img_linear_window[1] - img_linear_window[0]
        )

    return norm_img
