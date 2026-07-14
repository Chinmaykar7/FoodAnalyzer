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
    image : numpy.ndarray
        Loaded image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def print_image_info(image):
    """
    Print useful information about the image.
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

def resize_image(image, max_dimension=1500):
    """
    Resize an image while preserving aspect ratio.

    The longest side of the image is resized to
    max_dimension. Smaller images are left unchanged.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    max_dimension : int
        Maximum size of the longest image dimension.

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

    # Don't enlarge smaller images
    if longest_side <= max_dimension:
        print("\nResize Information")
        print("-" * 40)
        print("Image already within size limit.")
        print(f"Current Size : {width} x {height}")
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

def save_image(image, output_path):
    """
    Save image to disk.
    """

    cv2.imwrite(str(output_path), image)

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    image_path = project_root / "images" / "raw" / "test.jpg"

    # Load image
    original_image = load_image(str(image_path))

    print("Before resizing")
    print_image_info(original_image)

    # Resize image
    resized_image = resize_image(original_image)

    print("\nAfter resizing")
    print_image_info(resized_image)

    # Save resized image
    output_path = project_root / "images" / "processed" / "resized.jpg"

    save_image(resized_image, output_path)
