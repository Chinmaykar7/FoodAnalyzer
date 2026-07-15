import cv2


def convert_to_lab(image):
    """
    Convert a BGR image to LAB color space.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def split_lab_channels(lab_image):
    """
    Split a LAB image into its individual channels.

    Returns
    -------
    tuple
        (L, A, B)
    """

    return cv2.split(lab_image)


def apply_clahe(
    l_channel,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
):
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
        tileGridSize=tile_grid_size
    )

    enhanced_l = clahe.apply(l_channel)

    return enhanced_l


def merge_lab_channels(
    l_channel,
    a_channel,
    b_channel
):
    """
    Merge the LAB channels into a single LAB image.
    """

    return cv2.merge(
        (l_channel, a_channel, b_channel)
    )


def convert_lab_to_bgr(lab_image):
    """
    Convert LAB image back to BGR.
    """

    return cv2.cvtColor(
        lab_image,
        cv2.COLOR_LAB2BGR
    )