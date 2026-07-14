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


def resize_image(image, max_dimension: int = 1500):
    """
    Resize an image while preserving aspect ratio.

    The longest side is resized to `max_dimension`.
    Images already smaller than this limit are left unchanged.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    max_dimension : int
        Maximum allowed size of the longest image dimension.

    Returns
    -------
    numpy.ndarray
        Resized image.
    """

    height, width = image.shape[:2]

    longest_side = max(height, width)

    print("\nResize Information")
    print("-" * 40)
    print(f"Original Size : {width} x {height}")
    print(f"Longest Side  : {longest_side}")

    if longest_side <= max_dimension:
        print("Image already within size limit.")
        print("-" * 40)
        return image

    scale = max_dimension / longest_side

    new_width = int(width * scale)
    new_height = int(height * scale)

    print(f"Scale Factor  : {scale:.3f}")
    print(f"New Size      : {new_width} x {new_height}")
    print("-" * 40)

    resized_image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized_image


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


def save_image(image, output_path: Path):
    """
    Save an image to disk.

    Parameters
    ----------
    image : numpy.ndarray

    output_path : pathlib.Path
    """

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise IOError(
            f"Failed to save image to {output_path}"
        )

    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":

    # -------------------------------------------------
    # Project paths
    # -------------------------------------------------

    project_root = Path(__file__).resolve().parent.parent

    image_path = (
        project_root
        / "images"
        / "raw"
        / "test.jpg"
    )

    processed_dir = (
        project_root
        / "images"
        / "processed"
    )

    # Create processed directory if it doesn't exist
    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Load Image
    # -------------------------------------------------

    original_image = load_image(str(image_path))

    print("\nOriginal Image")
    print_image_info(original_image)

    # -------------------------------------------------
    # Resize
    # -------------------------------------------------

    resized_image = resize_image(original_image)

    print("\nResized Image")
    print_image_info(resized_image)

    save_image(
        resized_image,
        processed_dir / "resized.jpg"
    )

    # -------------------------------------------------
    # LAB Conversion
    # -------------------------------------------------

    lab_image = convert_to_lab(resized_image)

    l_channel, a_channel, b_channel = split_lab_channels(
        lab_image
    )

    # -------------------------------------------------
    # Save LAB Channels
    # -------------------------------------------------

    save_image(
        l_channel,
        processed_dir / "l_channel.jpg"
    )

    save_image(
        a_channel,
        processed_dir / "a_channel.jpg"
    )

    save_image(
        b_channel,
        processed_dir / "b_channel.jpg"
    )

    print("\nPreprocessing stage completed successfully.")