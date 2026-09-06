import json

from config import (
    PROJECT_ROOT,
    RAW_IMAGES_FOLDER,
    RAW_IMAGE_NAME,
)

from preprocessing.pipeline import (
    run_preprocessing_pipeline,
)

from extraction.extractor import (
    IngredientExtractor,
)

from parsing.parser import (
    FoodLabelParser,
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
    # Stage 1 - Image Preprocessing
    # --------------------------------------------------

    enhanced_image_path = run_preprocessing_pipeline(
        raw_image_path
    )

    # --------------------------------------------------
    # Stage 2 - Food Label Extraction
    # --------------------------------------------------

    food_label_extractor = IngredientExtractor()

    label_text = food_label_extractor.extract(
        enhanced_image_path
    )

    # --------------------------------------------------
    # Stage 3 - Food Label Parsing
    # --------------------------------------------------

    food_label_parser = FoodLabelParser()

    parsed_food_data = food_label_parser.parse(
        label_text
    )

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    print("\n========== Structured Food Data ==========\n")
    print(json.dumps(parsed_food_data, indent=4))


if __name__ == "__main__":
    main()