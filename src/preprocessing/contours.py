import cv2
import numpy as np


def find_contours(binary_image: np.ndarray) -> list:
    """
    Detect external contours in a binary image.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Binary image.

    Returns
    -------
    list
        List of detected contours.
    """

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    print("\nContour Information")
    print("-" * 40)
    print(f"Contours Found : {len(contours)}")
    print("-" * 40)

    return contours