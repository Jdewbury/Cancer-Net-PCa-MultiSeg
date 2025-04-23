import numpy as np
from pathlib import Path
import SimpleITK as sitk


def list_image_paths(root_dir: Path, modality: str = "cdis") -> list[Path]:
    """Gather all image paths in directory for specified modality.

    Args:
        root_dir: directory to search for image files
        modality: desired modality of interest

    Returns:
        Sorted list of image file paths.
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
