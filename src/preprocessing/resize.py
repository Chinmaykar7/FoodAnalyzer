import cv2


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