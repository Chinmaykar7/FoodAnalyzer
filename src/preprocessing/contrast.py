import cv2
import numpy as np

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE


def convert_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to LAB color space.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image.

    Returns
    -------
    numpy.ndarray
        LAB image.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def split_lab_channels(lab_image: np.ndarray) -> tuple:
    """
    Split a LAB image into its individual channels.

    Returns
    -------
    tuple
        (L, A, B) channels.
    """

    return cv2.split(lab_image)


def apply_clahe(
    l_channel: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the L (Lightness) channel.

    Parameters
    ----------
    l_channel : numpy.ndarray
        Lightness channel from LAB image.

    clip_limit : float
        Limits contrast amplification.

    tile_grid_size : tuple
        Size of the local grid used by CLAHE.

    Returns
    -------
    numpy.ndarray
        Enhanced L channel.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    enhanced_l = clahe.apply(l_channel)

    return enhanced_l


def merge_lab_channels(
    l_channel: np.ndarray,
    a_channel: np.ndarray,
    b_channel: np.ndarray,
) -> np.ndarray:
    """
    Merge the LAB channels into a single LAB image.
    """

    return cv2.merge((l_channel, a_channel, b_channel))


def convert_lab_to_bgr(lab_image: np.ndarray) -> np.ndarray:
    """
    Convert LAB image back to BGR.
    """

    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)