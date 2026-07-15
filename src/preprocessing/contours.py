import cv2
import numpy as np


def find_contours(binary_image):
    """
    Detect external contours in a binary image.

    Parameters
    ----------
    binary_image : numpy.ndarray

    Returns
    -------
    list
        List of detected contours.
    """

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print("\nContour Information")
    print("-" * 40)
    print(f"Contours Found : {len(contours)}")
    print("-" * 40)

    return contours


def filter_contours_by_area(
    contours,
    min_area: float = 100,
    max_area: float = None
):
    """
    Filter contours by area.

    Parameters
    ----------
    contours : list
        List of contours.
    min_area : float
        Minimum contour area.
    max_area : float or None
        Maximum contour area (None = no limit).

    Returns
    -------
    list
        Filtered contours.
    """

    filtered = [
        c for c in contours
        if cv2.contourArea(c) >= min_area
        and (max_area is None or cv2.contourArea(c) <= max_area)
    ]

    print(f"Filtered contours: {len(contours)} -> {len(filtered)} (area >= {min_area})")

    return filtered


def draw_contours(
    image,
    contours,
    color: tuple = (0, 255, 0),
    thickness: int = 2
):
    """
    Draw contours on an image.

    Parameters
    ----------
    image : numpy.ndarray
        Image to draw on (BGR).
    contours : list
        List of contours.
    color : tuple
        BGR color.
    thickness : int
        Line thickness.

    Returns
    -------
    numpy.ndarray
        Image with contours drawn.
    """

    output = image.copy()
    cv2.drawContours(output, contours, -1, color, thickness)
    return output


def get_bounding_rectangles(contours):
    """
    Get bounding rectangles for contours.

    Parameters
    ----------
    contours : list

    Returns
    -------
    list
        List of (x, y, w, h) tuples.
    """

    return [cv2.boundingRect(c) for c in contours]


def get_contour_centroids(contours):
    """
    Calculate centroids of contours.

    Parameters
    ----------
    contours : list

    Returns
    -------
    list
        List of (cx, cy) tuples.
    """

    centroids = []
    for c in contours:
        m = cv2.moments(c)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            centroids.append((cx, cy))
        else:
            centroids.append((0, 0))
    return centroids