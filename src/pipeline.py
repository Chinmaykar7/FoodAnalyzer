from pathlib import Path

from preprocessing.io import (
    load_image,
    save_image,
    print_image_info,
)

from preprocessing.resize import resize_image

from preprocessing.contrast import (
    convert_to_lab,
    split_lab_channels,
    apply_clahe,
    merge_lab_channels,
    convert_lab_to_bgr,
)

from preprocessing.threshold import create_binary_image

from preprocessing.morphology import apply_morphological_closing

from preprocessing.contours import find_contours


def main():
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


    # -------------------------------------------------
    # LAB Conversion
    # -------------------------------------------------

    lab_image = convert_to_lab(resized_image)

    l_channel, a_channel, b_channel = split_lab_channels(
        lab_image
    )

    # -------------------------------------------------
    # Apply CLAHE to L channel
    # -------------------------------------------------

    enhanced_l_channel = apply_clahe(l_channel)

    # -------------------------------------------------
    # Merge channels
    # -------------------------------------------------

    enhanced_lab_image = merge_lab_channels(
        enhanced_l_channel,
        a_channel,
        b_channel
    )

    # -------------------------------------------------
    # Convert back to BGR
    # -------------------------------------------------

    enhanced_image = convert_lab_to_bgr(
        enhanced_lab_image
    )

    # -------------------------------------------------
    # Binary Image
    # -------------------------------------------------

    binary_image = create_binary_image(
        enhanced_image
    )

    # -------------------------------------------------
    # Morphological Closing
    # -------------------------------------------------

    closed_image = apply_morphological_closing(
        binary_image
    )

    # -------------------------------------------------
    # Find Contours
    # -------------------------------------------------

    contours = find_contours(
        closed_image
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    save_image(
        resized_image,
        processed_dir / "resized.jpg"
    )

    save_image(
        enhanced_image,
        processed_dir / "enhanced.jpg"
    )

    save_image(
        binary_image,
        processed_dir / "binary.jpg"
    )

    save_image(
        closed_image,
        processed_dir / "closed.jpg"
    )

    print("\nCLAHE enhancement completed successfully.")


if __name__ == "__main__":
    main()