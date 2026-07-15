import cv2
import numpy as np


def create_binary_image(enhanced_image: np.ndarray) -> np.ndarray:
    """
    Convert an enhanced BGR image into a binary image
    using Otsu's thresholding.

    Parameters
    ----------
    enhanced_image : numpy.ndarray
        Enhanced BGR image.

    Returns
    -------
    numpy.ndarray
        Binary image.
    """

    gray_image = cv2.cvtColor(
        enhanced_image,
        cv2.COLOR_BGR2GRAY,
    )

    threshold_value, binary_image = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    print("\nThreshold Information")
    print("-" * 40)
    print(f"Otsu Threshold : {threshold_value:.2f}")
    print("-" * 40)

    return binary_image