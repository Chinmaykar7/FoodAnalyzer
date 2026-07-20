from pathlib import Path

from config import (
    PROJECT_ROOT,
    PROCESSED_IMAGES_FOLDER,
    MAX_IMAGE_DIMENSION,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
)

from preprocessing.io import (
    load_image,
    save_image,
    print_image_info,
)

from preprocessing.resize import (
    resize_image,
)

from preprocessing.contrast import (
    convert_to_lab,
    split_lab_channels,
    apply_clahe,
    merge_lab_channels,
    convert_lab_to_bgr,
)


def run_preprocessing_pipeline(image_path: Path) -> Path:
    """
    Run the complete image preprocessing pipeline.

    Steps:
        1. Load image
        2. Resize image
        3. Enhance contrast using CLAHE in LAB color space
        4. Save resized and enhanced images

    Args:
        image_path:
            Path to the raw input image.

    Returns:
        Path to the enhanced image.
    """

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    image_path = Path(image_path).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    processed_dir = PROJECT_ROOT / PROCESSED_IMAGES_FOLDER
    processed_dir.mkdir(parents=True, exist_ok=True)

    resized_image_path = processed_dir / "resized.jpg"
    enhanced_image_path = processed_dir / "enhanced.jpg"

    # --------------------------------------------------
    # Load Image
    # --------------------------------------------------

    original_image = load_image(str(image_path))

    print("\nOriginal Image")
    print_image_info(original_image)

    # --------------------------------------------------
    # Resize
    # --------------------------------------------------

    resized_image = resize_image(
        original_image,
        max_dimension=MAX_IMAGE_DIMENSION,
    )

    print("\nResized Image")
    print_image_info(resized_image)

    # --------------------------------------------------
    # LAB Conversion
    # --------------------------------------------------

    lab_image = convert_to_lab(resized_image)

    l_channel, a_channel, b_channel = split_lab_channels(lab_image)

    # --------------------------------------------------
    # Apply CLAHE
    # --------------------------------------------------

    enhanced_l_channel = apply_clahe(
        l_channel,
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=CLAHE_TILE_GRID_SIZE,
    )

    # --------------------------------------------------
    # Merge Channels
    # --------------------------------------------------

    enhanced_lab_image = merge_lab_channels(
        enhanced_l_channel,
        a_channel,
        b_channel,
    )

    # --------------------------------------------------
    # Convert back to BGR
    # --------------------------------------------------

    enhanced_image = convert_lab_to_bgr(
        enhanced_lab_image,
    )

    # --------------------------------------------------
    # Save Images
    # --------------------------------------------------

    save_image(
        resized_image,
        resized_image_path,
    )

    save_image(
        enhanced_image,
        enhanced_image_path,
    )

    print("\n✅ Preprocessing completed successfully.")

    return enhanced_image_path