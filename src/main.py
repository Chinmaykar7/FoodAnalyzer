from config import (
    PROJECT_ROOT,
    RAW_IMAGES_FOLDER,
    RAW_IMAGE_NAME,
)

from preprocessing.pipeline import (
    run_preprocessing_pipeline,
)

from models.extractor import (
    IngredientExtractor,
)


def main() -> None:
    """
    Entry point for the Food Analyzer pipeline.
    """

    # --------------------------------------------------
    # Raw input image
    # --------------------------------------------------

    raw_image_path = (
        PROJECT_ROOT
        / RAW_IMAGES_FOLDER
        / RAW_IMAGE_NAME
    )

    # --------------------------------------------------
    # Stage 1 - Preprocessing
    # --------------------------------------------------

    enhanced_image_path = run_preprocessing_pipeline(
        raw_image_path
    )

    # --------------------------------------------------
    # Stage 2 - Ingredient Extraction
    # --------------------------------------------------

    extractor = IngredientExtractor()

    ingredients = extractor.extract(
        enhanced_image_path
    )

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    print("\n========== Extracted Ingredients ==========\n")
    print(ingredients)


if __name__ == "__main__":
    main()