from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_IMAGES_FOLDER = "images/raw"
PROCESSED_IMAGES_FOLDER = "images/processed"
RAW_IMAGE_NAME = "test.jpg"

MAX_IMAGE_DIMENSION = 1500
RESIZE_MAX_DIMENSION = MAX_IMAGE_DIMENSION

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

MORPH_KERNEL_SIZE = (3, 3)