from os import makedirs
from os.path import exists, join

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from tqdm import tqdm

from utils.consts import details_df, preprocessed_dataset_path


def partition_img(
    img_list: list[cv2.Mat], horizontal_patch: int = 64, vertical_patch: int = 64
) -> np.ndarray:
    """Partition a list of images symmetrically
    according to the provided patch dimensions
    and perform vertical projection to the patches

    Args:
        img_list (list[cv2.Mat]): List of images to partition
        horizontal_patch (int, optional): Width of each patch. Defaults to 64.
        vertical_patch (int, optional): Height of each patch. Defaults to 64.

    Returns:
        np.ndarray: Numpy list of concatenated vertical projected patches
    """
    patches = []
    for img in img_list:
        for x in range(0, img.shape[0] + 1 - horizontal_patch, horizontal_patch):
            for y in range(0, img.shape[1] + 1 - vertical_patch, vertical_patch):
                patch = img[x : x + horizontal_patch, y : y + vertical_patch]
                vector = np.sum(patch, axis=1)
                patches.append(vector)

    return np.array(patches)


def generate_schmid_kernel(kernel_size: int, tau: int, sigma: int) -> np.ndarray:
    """Generate a kernel according to Cordelia Schmid's
    paper (inria-00548274) for a Gabor-like filter

    Args:
        kernel_size (int): Applied kernel size
        tau (int): Tau value
        sigma (int): Sigma value

    Returns:
        np.array: Normalized kernel
    """
    end_point = int((kernel_size - 1) / 2)
    x = np.linspace(-end_point, end_point, kernel_size)
    y = np.linspace(-end_point, end_point, kernel_size)

    [xv, yv] = np.meshgrid(x, y)

    # x^2 + y^2
    mul = (xv**2) + (yv**2)

    sr = np.sqrt(mul)

    f = (np.cos(sr * np.pi * tau) / sigma) * (np.exp(-(mul) / (2 * sigma**2)))

    # Normalization
    f -= np.mean(f)
    f /= np.sum(np.abs(f))

    return f


def apply_schmid_filter(img: cv2.Mat) -> list[cv2.Mat]:
    """Applies set of schmid filters to the image

    Args:
        img (cv2.Mat): Original image

    Returns:
        list[cv2.Mat]: List of filtered images
    """
    schmid_kernel_size = 49

    schmid_filter_parameter_values = [
        [2, 1],
        [4, 2],
        [6, 2],
        [8, 2],
        [10, 2],
    ]
    filtered_imgs_list = []
    for parameter_value in schmid_filter_parameter_values:
        kernel = generate_schmid_kernel(schmid_kernel_size, *parameter_value)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        filtered_imgs_list.append(filtered_img)
    return filtered_imgs_list


def apply_gabor_filter(img: cv2.Mat) -> list[cv2.Mat]:
    """Applies set of gabor filters to the image

    Args:
        img (cv2.Mat): Original image

    Returns:
        list[cv2.Mat]: List of filtered images
    """
    gabor_kernel_size = (15, 15)

    gabor_filter_parameter_values = [
        [
            2,
            0,
            4,
            0.3,
        ],
        [
            1,
            0,
            4,
            0.4,
        ],
        [
            2,
            np.pi / 2,
            8,
            0.3,
        ],
        [
            1,
            np.pi / 2,
            8,
            0.4,
        ],
    ]
    filtered_img_list = []
    for parameter_value in gabor_filter_parameter_values:
        kernel = cv2.getGaborKernel(gabor_kernel_size, *parameter_value, 0)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        filtered_img_list.append(filtered_img)
    return filtered_img_list


def apply_hog_descriptor(img: cv2.Mat) -> list[cv2.Mat]:
    """Applies HOG descriptor with different parameters

    Args:
        img (cv2.Mat): Original

    Returns:
        list[cv2.Mat]: List of filtered images
    """
    hog_decriptor_parameter_values = [
        [8, (16, 16)],
        [8, (32, 32)],
        [16, (16, 16)],
        [15, (32, 32)],
    ]
    filtered_imgs_list = []
    for parameter_value in hog_decriptor_parameter_values:
        _, filtered_img = hog(
            img,
            orientations=parameter_value[0],
            pixels_per_cell=parameter_value[1],
            cells_per_block=(1, 1),
            visualize=True,
        )
        filtered_imgs_list.append(filtered_img)

    return filtered_imgs_list


def load_preprocessed_data(descriptor: str) -> np.ndarray:
    """Load preprocessed data from the project files
    according to specific descriptor

    Args:
        descriptor (str): Name of preprocessing descriptor

    Returns:
        np.ndarray: Preprocessed data
    """
    vectors_list = []
    for _, row in tqdm(
        details_df.iterrows(),
        desc=f"Loading the {descriptor} preprocessed data",
        colour="cyan",
        total=len(details_df),
    ):
        name = row["name"]
        ary_path = join(
            preprocessed_dataset_path,
            descriptor,
            row["class"],
            f"{name}.npy",
        )

        np_array = np.load(ary_path)

        vectors_list.append(np_array)

    vectors_list = np.array(vectors_list)
    return vectors_list


def hist_match(query_hist: np.ndarray, dataset_img_hist: np.ndarray) -> float:
    """Calculates histogram similarity between
    2 histograms

    Args:
        query_hist (np.ndarray): First histogram in comparison
        dataset_img_hist (np.ndarray): Second histogram in comparison

    Returns:
        float: Comparsion result (0:1)
    """
    result = 0.0
    for i in range(query_hist.size):
        if query_hist[i] <= dataset_img_hist[i]:
            result = result + query_hist[i]
        else:
            result = result + dataset_img_hist[i]
    return result


def preprocess_img(img: cv2.Mat, descriptor: str) -> list[cv2.Mat]:
    """Perform set of preprocessing steps according
    to different descriptors

    Args:
        img (cv2.Mat): Original image
        descriptor (str): Descriptor name

    Raises:
        ValueError: Chosen descriptor is not exist

    Returns:
        list[cv2.Mat]: List of filtered images
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if descriptor == "original":
        # Apply Gabor filter
        filtered_gabor = apply_gabor_filter(img)
        # Apply Schmid filter
        filtered_shmid = apply_schmid_filter(img)
        # Merge the 2 filtered Images
        filtered_img = filtered_gabor + filtered_shmid
        return filtered_img
    elif descriptor == "hog":
        filtered_img = apply_hog_descriptor(img)
        return filtered_img

    else:
        raise ValueError("The value you entered is not exist")


def create_dirs(*paths) -> str:
    """Perform safe joining and making of
    directory

    Returns:
        str: Path of the generated directory
    """
    dir_path = join(*paths)
    if not exists(dir_path):
        makedirs(dir_path)

    return dir_path
