import cv2


def apply_morphological_closing(
    binary_image,
    kernel_size: tuple = (3, 3)
):
    """
    Apply morphological closing to connect nearby
    text regions.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Binary image.

    kernel_size : tuple
        Size of the rectangular structuring element.

    Returns
    -------
    numpy.ndarray
        Morphologically processed image.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        kernel_size
    )

    closed_image = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closed_image