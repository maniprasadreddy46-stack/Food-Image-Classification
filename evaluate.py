import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "food_classifier.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.json"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)

REPORT_PATH = os.path.join(
    ASSETS_DIR,
    "classification_report.txt"
)

CM_PATH = os.path.join(
    ASSETS_DIR,
    "confusion_matrix.png"
)

SUMMARY_PATH = os.path.join(
    ASSETS_DIR,
    "evaluation_summary.txt"
)


os.makedirs(
    ASSETS_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("INDIAN FOOD IMAGE CLASSIFICATION - MODEL EVALUATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking files...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found:\n{CLASS_NAMES_PATH}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


print("Model found       :", MODEL_PATH)
print("Class names found :", CLASS_NAMES_PATH)
print("Test directory    :", TEST_DIR)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print("\nLoading class names...")

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_names = json.load(f)


print(
    "Number of classes:",
    len(class_names)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# MODEL OUTPUT CHECK
# ============================================================

model_output_classes = model.output_shape[-1]

print(
    "Model output classes:",
    model_output_classes
)

if model_output_classes != len(class_names):

    raise ValueError(
        "\nMODEL / CLASS NAME MISMATCH\n"
        f"Model outputs : {model_output_classes}\n"
        f"Class names   : {len(class_names)}\n"
    )


# ============================================================
# TEST DATA GENERATOR
# ============================================================

print("\nLoading test dataset...")

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# ============================================================
# IMPORTANT:
# categorical = 80-dimensional labels
# ============================================================

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print()
print("=" * 70)
print("TEST DATASET INFORMATION")
print("=" * 70)

print(
    "Test images       :",
    test_generator.samples
)

print(
    "Number of classes :",
    len(test_generator.class_indices)
)


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

generator_class_names = [
    None
] * len(test_generator.class_indices)


for class_name, index in test_generator.class_indices.items():

    generator_class_names[index] = class_name


if generator_class_names != class_names:

    print()
    print("WARNING: Class order differs between")
    print("test generator and class_names.json")

    print()
    print("Generator classes:")
    print(generator_class_names)

    print()
    print("Saved classes:")
    print(class_names)

else:

    print(
        "\nClass order verified successfully."
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

print()
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    test_generator,
    verbose=1
)


# ============================================================
# PREDICTED LABELS
# ============================================================

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# TRUE LABELS
# ============================================================

true_labels = test_generator.classes


# ============================================================
# BASIC ACCURACY
# ============================================================

calculated_accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Test Loss              : "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy          : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Calculated Accuracy    : "
    f"{calculated_accuracy * 100:.2f}%"
)

print(
    f"Total Test Images      : "
    f"{len(true_labels)}"
)

print(
    f"Total Food Classes     : "
    f"{len(class_names)}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(

    true_labels,

    predicted_labels,

    labels=np.arange(
        len(class_names)
    ),

    target_names=class_names,

    zero_division=0
)


print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "INDIAN FOOD IMAGE CLASSIFICATION\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Number of classes : "
        f"{len(class_names)}\n"
    )

    f.write(
        f"Test images       : "
        f"{len(true_labels)}\n"
    )

    f.write(
        f"Test loss         : "
        f"{test_loss:.4f}\n"
    )

    f.write(
        f"Test accuracy     : "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CREATING CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(
        len(class_names)
    )
)


plt.figure(
    figsize=(22, 20)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Indian Food Classification - Confusion Matrix",
    fontsize=16
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=90,
    fontsize=7
)

plt.yticks(
    tick_marks,
    class_names,
    fontsize=7
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()

plt.savefig(
    CM_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "Confusion matrix saved:"
)

print(
    CM_PATH
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print()
print("=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)


per_class_accuracy = []


for i, class_name in enumerate(
    class_names
):

    total = np.sum(
        true_labels == i
    )

    correct = np.sum(
        (true_labels == i)
        &
        (predicted_labels == i)
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    per_class_accuracy.append(
        (
            class_name,
            accuracy,
            int(total),
            int(correct)
        )
    )


# Sort from lowest to highest

sorted_per_class = sorted(
    per_class_accuracy,
    key=lambda x: x[1]
)


for (
    class_name,
    accuracy,
    total,
    correct
) in sorted_per_class:

    print(
        f"{class_name:30s} "
        f"{accuracy * 100:6.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# BIRYANI SPECIFIC CHECK
# ============================================================

print()
print("=" * 70)
print("BIRYANI CHECK")
print("=" * 70)


if "biryani" in class_names:

    biryani_index = class_names.index(
        "biryani"
    )

    biryani_total = np.sum(
        true_labels == biryani_index
    )

    biryani_correct = np.sum(
        (true_labels == biryani_index)
        &
        (predicted_labels == biryani_index)
    )

    if biryani_total > 0:

        biryani_accuracy = (
            biryani_correct
            /
            biryani_total
        )

    else:

        biryani_accuracy = 0.0


    print(
        "Biryani class index :",
        biryani_index
    )

    print(
        "Biryani test images :",
        biryani_total
    )

    print(
        "Biryani correct     :",
        biryani_correct
    )

    print(
        f"Biryani accuracy    : "
        f"{biryani_accuracy * 100:.2f}%"
    )

else:

    biryani_index = None
    biryani_total = 0
    biryani_correct = 0
    biryani_accuracy = 0.0

    print(
        "Biryani class not found!"
    )


# ============================================================
# BEST AND WORST CLASSES
# ============================================================

valid_classes = [
    item
    for item in per_class_accuracy
    if item[2] > 0
]


if valid_classes:

    best_class = max(
        valid_classes,
        key=lambda x: x[1]
    )

    worst_class = min(
        valid_classes,
        key=lambda x: x[1]
    )

else:

    best_class = (
        "N/A",
        0.0,
        0,
        0
    )

    worst_class = (
        "N/A",
        0.0,
        0,
        0
    )


# ============================================================
# SAVE EVALUATION SUMMARY
# ============================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FOOD IMAGE CLASSIFICATION - "
        "EVALUATION SUMMARY\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Number of classes : "
        f"{len(class_names)}\n"
    )

    f.write(
        f"Test images       : "
        f"{len(true_labels)}\n"
    )

    f.write(
        f"Test loss         : "
        f"{test_loss:.4f}\n"
    )

    f.write(
        f"Test accuracy     : "
        f"{test_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Calculated accuracy: "
        f"{calculated_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "BIRYANI RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Biryani index     : "
        f"{biryani_index}\n"
    )

    f.write(
        f"Biryani test images: "
        f"{biryani_total}\n"
    )

    f.write(
        f"Biryani correct   : "
        f"{biryani_correct}\n"
    )

    f.write(
        f"Biryani accuracy  : "
        f"{biryani_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "BEST CLASS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"{best_class[0]} : "
        f"{best_class[1] * 100:.2f}% "
        f"({best_class[3]}/{best_class[2]})\n\n"
    )

    f.write(
        "WORST CLASS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"{worst_class[0]} : "
        f"{worst_class[1] * 100:.2f}% "
        f"({worst_class[3]}/{worst_class[2]})\n\n"
    )

    f.write(
        "PER-CLASS ACCURACY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for (
        class_name,
        accuracy,
        total,
        correct
    ) in sorted_per_class:

        f.write(
            f"{class_name:30s} "
            f"{accuracy * 100:6.2f}% "
            f"({correct}/{total})\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 70)

print()
print("Files created:")

print(
    f"1. {REPORT_PATH}"
)

print(
    f"2. {CM_PATH}"
)

print(
    f"3. {SUMMARY_PATH}"
)

print()
print("=" * 70)
print("FINAL TEST ACCURACY:")
print(
    f"{test_accuracy * 100:.2f}%"
)
print("=" * 70)

print()
print("BIRYANI TEST ACCURACY:")

print(
    f"{biryani_accuracy * 100:.2f}%"
)

print("=" * 70)