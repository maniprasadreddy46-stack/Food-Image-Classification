"""
prepare_dataset.py
------------------
Prepares the 80-class Indian Food dataset.

Input:
    dataset/raw/<food_class>/*.jpg

Output:
    dataset/train/<food_class>/
    dataset/validation/<food_class>/
    dataset/test/<food_class>/

Split:
    68% training
    16% validation
    16% testing
"""

from pathlib import Path
import shutil
import random
import json

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "dataset" / "raw"
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "validation"
TEST_DIR = BASE_DIR / "dataset" / "test"

MODEL_DIR = BASE_DIR / "model"

SEED = 42

TRAIN_RATIO = 0.68
VAL_RATIO = 0.16
TEST_RATIO = 0.16

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

random.seed(SEED)


# ============================================================
# FUNCTIONS
# ============================================================

def clean_directory(directory):
    """
    Remove an existing directory and recreate it.
    """

    if directory.exists():
        print(f"Removing old directory: {directory}")
        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)


def get_images(class_dir):
    """
    Get valid image files from a class directory.
    """

    images = []

    for file in class_dir.iterdir():

        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)

    return images


# ============================================================
# CHECK DATASET
# ============================================================

if not RAW_DIR.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{RAW_DIR}\n\n"
        "Make sure the 80 food classes are inside dataset/raw/"
    )


class_dirs = sorted(
    [
        directory
        for directory in RAW_DIR.iterdir()
        if directory.is_dir()
    ],
    key=lambda x: x.name.lower()
)


print("\n" + "=" * 70)
print("INDIAN FOOD DATASET PREPARATION")
print("=" * 70)

print(f"\nDataset location : {RAW_DIR}")
print(f"Classes found    : {len(class_dirs)}")

if len(class_dirs) != 80:

    print(
        f"\nWARNING: Expected 80 classes but found {len(class_dirs)}."
    )

    answer = input(
        "Continue anyway? Type YES to continue: "
    ).strip().upper()

    if answer != "YES":
        raise SystemExit("Dataset preparation cancelled.")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

print("\nCleaning previous prepared dataset...")

clean_directory(TRAIN_DIR)
clean_directory(VAL_DIR)
clean_directory(TEST_DIR)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_images = 0
total_train = 0
total_val = 0
total_test = 0

class_names = []


print("\nPreparing classes...\n")


for index, class_dir in enumerate(class_dirs, start=1):

    class_name = class_dir.name

    class_names.append(class_name)

    images = get_images(class_dir)

    random.shuffle(images)

    total = len(images)

    if total == 0:

        print(
            f"[{index:02d}/{len(class_dirs)}] "
            f"{class_name}: NO IMAGES"
        )

        continue


    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    train_count = int(total * TRAIN_RATIO)

    val_count = int(total * VAL_RATIO)

    test_count = total - train_count - val_count


    train_images = images[:train_count]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]


    # --------------------------------------------------------
    # Create class folders
    # --------------------------------------------------------

    train_class_dir = TRAIN_DIR / class_name
    val_class_dir = VAL_DIR / class_name
    test_class_dir = TEST_DIR / class_name

    train_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    val_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Copy training images
    # --------------------------------------------------------

    for image in train_images:

        shutil.copy2(
            image,
            train_class_dir / image.name
        )


    # --------------------------------------------------------
    # Copy validation images
    # --------------------------------------------------------

    for image in val_images:

        shutil.copy2(
            image,
            val_class_dir / image.name
        )


    # --------------------------------------------------------
    # Copy testing images
    # --------------------------------------------------------

    for image in test_images:

        shutil.copy2(
            image,
            test_class_dir / image.name
        )


    total_images += total
    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)


    print(
        f"[{index:02d}/{len(class_dirs)}] "
        f"{class_name:<25} "
        f"Total={total:<3} "
        f"Train={len(train_images):<3} "
        f"Val={len(val_images):<3} "
        f"Test={len(test_images):<3}"
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = MODEL_DIR / "class_names.json"

with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SAVE DATASET INFORMATION
# ============================================================

dataset_info = {

    "number_of_classes": len(class_names),

    "total_images": total_images,

    "training_images": total_train,

    "validation_images": total_val,

    "testing_images": total_test,

    "train_ratio": TRAIN_RATIO,

    "validation_ratio": VAL_RATIO,

    "test_ratio": TEST_RATIO,

    "random_seed": SEED,

    "classes": class_names
}


info_path = MODEL_DIR / "dataset_info.json"

with open(
    info_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        dataset_info,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETED")
print("=" * 70)

print(f"\nNumber of classes : {len(class_names)}")
print(f"Total images      : {total_images}")
print(f"Training images   : {total_train}")
print(f"Validation images : {total_val}")
print(f"Testing images    : {total_test}")

print("\nClass names saved to:")
print(class_names_path)

print("\nPrepared dataset:")

print(f"Train      : {TRAIN_DIR}")
print(f"Validation : {VAL_DIR}")
print(f"Test       : {TEST_DIR}")

print("\nNext step:")
print("python train.py")

print("=" * 70)