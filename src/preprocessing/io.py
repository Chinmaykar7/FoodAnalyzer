from pathlib import Path
import cv2


def load_image(image_path: str):
    """
    Load an image from disk.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    numpy.ndarray
        Loaded image.

    Raises
    ------
    FileNotFoundError
        If the image cannot be loaded.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def save_image(image, output_path: Path):
    """
    Save an image to disk.

    Parameters
    ----------
    image : numpy.ndarray

    output_path : pathlib.Path
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise IOError(
            f"Failed to save image to {output_path}"
        )

    print(f"✓ Saved: {output_path}")


def print_image_info(image):
    """
    Print useful information about an image.
    """

    print("=" * 40)
    print("Image Information")
    print("=" * 40)

    print(f"Shape      : {image.shape}")
    print(f"Height     : {image.shape[0]} px")
    print(f"Width      : {image.shape[1]} px")
    print(f"Channels   : {image.shape[2]}")
    print(f"Data Type  : {image.dtype}")

    print("=" * 40)