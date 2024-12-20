import glob
import numpy as np
import nibabel as nib

def list_nii_paths(directory):
    """Generator function to iterate over all nii files in a given directory.

    Args:
        directory: Directory path to search for nii files.

    Returns:
        Sorted array of file paths for each nii file found.
    """
    file_paths = glob.glob(f'{directory}/**/*.nii', recursive=True)
    return np.array(sorted(file_paths))

def list_prostate_paths(directory):
    """Generator function to iterate over all prostate and lesion mask files in a given directory.

    Args:
        directory: Directory path to search for mask files.

    Returns:
        Sorted array of file paths for lesion and prostate mask files found.
    """
    lesion_paths = glob.glob(f'{directory}/**/lesion_mask.npy', recursive=True)
    prostate_paths = glob.glob(f'{directory}/**/prostate_mask.npy', recursive=True)
    return np.array([sorted(lesion_paths), sorted(prostate_paths)])

def list_npy_paths(directory, modality='cdis'):
    """Generator function to iterate over all npy files in a given directory for a specified modality.

    Args:
        directory: Directory path to search for nnpy files.
        modality: Desired modality of interest

    Returns:
        Sorted array of file paths for each npy file found.
    """
    file_paths = glob.glob(f'{directory}/**/{modality.upper()}.npy', recursive=True)
    return np.array(sorted(file_paths))

def list_img_paths(directory, modality='cdis'):
    """Generator function to return image paths in a given directory for a specified modality.

    Args:
        directory: Directory path to search for nnpy files.
        modality: Desired modality of interest

    Returns:
        Sorted array of file paths for each image file found.
    """
    if modality in ['adc', 'dwi']:
        return list_npy_paths(directory, modality)
    elif modality == 'cdis':
        return list_nii_paths(directory)
    else:
        raise ValueError(f"Invalid modality name: {modality}. Choose from 'cdis', 'dwi', or 'adc'.")

def modality_to_numpy(directory, modality='cdis', channel_idx=0):
    """Load a modality image and convert it to a numpy array.

    Args:
        directory: Directory path to convert modality file to numpy array.

    Returns:
        Numpy array of type uint8.
    """
    if modality == 'adc':
        img_np = np.load(directory, allow_pickle=True)
        img = np.transpose(img_np, (2, 1, 0))
    elif modality == 'dwi':
        img_np = np.load(directory, allow_pickle=True)
        img = np.transpose(img_np, (3, 2, 1, 0))
        img = img[:, :, :, channel_idx]
    elif modality == 'cdis':
        img_nib = nib.load(directory).get_fdata()
        img_nib = np.nan_to_num(img_nib)
        img_np = np.array(img).astype(np.uint8)
        img_f = img_np.astype(float)
        img = img_f.astype(np.float32)
    else:
        raise ValueError(f"Invalid modality name: {modality}. Choose from 'cdis', 'dwi', or 'adc'.")

    img_linear_window = [img.min(), img.max()]
    img_clip = np.clip(img, *img_linear_window)
    norm_img = (img_clip - img_linear_window[0]) / (
        img_linear_window[1] - img_linear_window[0]
    )

    return norm_img