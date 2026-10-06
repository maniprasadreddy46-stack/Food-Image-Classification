import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 25

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "validation"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "food_classifier.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

ACCURACY_GRAPH = os.path.join(
    ASSETS_DIR,
    "training_accuracy_final.png"
)

LOSS_GRAPH = os.path.join(
    ASSETS_DIR,
    "training_loss_final.png"
)

CSV_LOG = os.path.join(
    ASSETS_DIR,
    "training_history_final.csv"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    ASSETS_DIR,
    exist_ok=True
)


# ============================================================
# RANDOM SEEDS
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("INDIAN FOOD IMAGE CLASSIFICATION")
print("FINAL TRAINING - EfficientNetB0")
print("=" * 75)


# ============================================================
# CHECK DATASET
# ============================================================

print()
print("Checking dataset...")


for folder in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"Dataset folder not found:\n{folder}"
        )


print("Training folder   :", TRAIN_DIR)
print("Validation folder :", VAL_DIR)
print("Test folder       :", TEST_DIR)


# ============================================================
# DATA AUGMENTATION
# ============================================================

print()
print("=" * 75)
print("CREATING DATA GENERATORS")
print("=" * 75)


# EfficientNetB0 in tf.keras already handles
# its input rescaling internally.
#
# Therefore DO NOT use MobileNetV2 preprocess_input here.

train_datagen = ImageDataGenerator(

    rotation_range=25,

    width_shift_range=0.15,

    height_shift_range=0.15,

    shear_range=0.15,

    zoom_range=0.20,

    horizontal_flip=True,

    brightness_range=[
        0.75,
        1.25
    ],

    channel_shift_range=10.0,

    fill_mode="nearest"
)


validation_datagen = ImageDataGenerator()


test_datagen = ImageDataGenerator()


# ============================================================
# TRAIN GENERATOR
# ============================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True,

    seed=SEED
)


# ============================================================
# VALIDATION GENERATOR
# ============================================================

validation_generator = validation_datagen.flow_from_directory(

    VAL_DIR,

    target_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# ============================================================
# TEST GENERATOR
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

NUM_CLASSES = len(
    train_generator.class_indices
)


print()
print("=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print(
    "Training images   :",
    train_generator.samples
)

print(
    "Validation images :",
    validation_generator.samples
)

print(
    "Test images       :",
    test_generator.samples
)

print(
    "Number of classes :",
    NUM_CLASSES
)


# ============================================================
# CHECK 80 CLASSES
# ============================================================

if NUM_CLASSES != 80:

    print()
    print(
        f"WARNING: Expected 80 classes, "
        f"but found {NUM_CLASSES}."
    )


# ============================================================
# CREATE CLASS NAMES
# ============================================================

class_indices = train_generator.class_indices

class_names = [
    None
] * NUM_CLASSES


for class_name, index in class_indices.items():

    class_names[index] = class_name


# ============================================================
# VERIFY VALIDATION CLASS ORDER
# ============================================================

validation_classes = [
    None
] * NUM_CLASSES


for class_name, index in (
    validation_generator.class_indices.items()
):

    validation_classes[index] = class_name


if class_names != validation_classes:

    raise ValueError(
        "Training and validation class "
        "orders do not match!"
    )


# ============================================================
# VERIFY TEST CLASS ORDER
# ============================================================

test_classes = [
    None
] * NUM_CLASSES


for class_name, index in (
    test_generator.class_indices.items()
):

    test_classes[index] = class_name


if class_names != test_classes:

    raise ValueError(
        "Training and test class "
        "orders do not match!"
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_names,
        f,
        indent=4,
        ensure_ascii=False
    )


print()
print(
    "Class names saved to:"
)

print(
    CLASS_NAMES_PATH
)


# ============================================================
# BIRYANI INFORMATION
# ============================================================

print()
print("=" * 75)
print("BIRYANI DATASET CHECK")
print("=" * 75)


if "biryani" in class_indices:

    biryani_index = class_indices[
        "biryani"
    ]

    print(
        "Biryani class index:",
        biryani_index
    )

    biryani_train_dir = os.path.join(
        TRAIN_DIR,
        "biryani"
    )

    biryani_val_dir = os.path.join(
        VAL_DIR,
        "biryani"
    )

    biryani_test_dir = os.path.join(
        TEST_DIR,
        "biryani"
    )

    def count_images(folder):

        if not os.path.exists(folder):

            return 0

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        )

        return len([
            f
            for f in os.listdir(folder)
            if f.lower().endswith(
                valid_extensions
            )
        ])


    biryani_train_count = count_images(
        biryani_train_dir
    )

    biryani_val_count = count_images(
        biryani_val_dir
    )

    biryani_test_count = count_images(
        biryani_test_dir
    )

    print(
        "Biryani training images   :",
        biryani_train_count
    )

    print(
        "Biryani validation images :",
        biryani_val_count
    )

    print(
        "Biryani test images       :",
        biryani_test_count
    )

else:

    raise ValueError(
        "Biryani class was not found!"
    )


# ============================================================
# LOAD EFFICIENTNETB0
# ============================================================

print()
print("=" * 75)
print("LOADING EFFICIENTNETB0")
print("=" * 75)


base_model = EfficientNetB0(

    weights="imagenet",

    include_top=False,

    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# ============================================================
# PHASE 1 - FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

x = base_model.output


x = GlobalAveragePooling2D()(x)


x = BatchNormalization()(x)


x = Dense(
    512,
    activation="relu"
)(x)


x = Dropout(
    0.45
)(x)


x = Dense(
    256,
    activation="relu"
)(x)


x = Dropout(
    0.30
)(x)


output = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# PHASE 1 COMPILE
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.0005
    ),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.05
    ),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print(
    "Model created successfully."
)

print(
    "Output classes:",
    model.output_shape[-1]
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=7,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=3,

    min_lr=1e-7,

    verbose=1
)


csv_logger = CSVLogger(

    CSV_LOG,

    append=False
)


# ============================================================
# PHASE 1 TRAINING
# ============================================================

print()
print("=" * 75)
print("PHASE 1 - TRANSFER LEARNING")
print("=" * 75)

print()
print(
    f"Training for up to "
    f"{INITIAL_EPOCHS} epochs..."
)


history1 = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=INITIAL_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ],

    verbose=1
)


# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print()
print("=" * 75)
print("PHASE 2 - FINE TUNING EFFICIENTNETB0")
print("=" * 75)


base_model.trainable = True


# Freeze most layers.
# Only the final portion will be fine-tuned.

for layer in base_model.layers[:-50]:

    layer.trainable = False


# Keep BatchNormalization layers frozen.
# This helps stability with a small dataset.

for layer in base_model.layers:

    if isinstance(
        layer,
        BatchNormalization
    ):

        layer.trainable = False


trainable_layers = sum(
    1
    for layer in model.layers
    if layer.trainable
)

frozen_layers = sum(
    1
    for layer in model.layers
    if not layer.trainable
)


print(
    "Trainable layers:",
    trainable_layers
)

print(
    "Frozen layers   :",
    frozen_layers
)


# ============================================================
# RECOMPILE FOR FINE TUNING
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.00002
    ),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.03
    ),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# PHASE 2 TRAINING
# ============================================================

history2 = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ],

    verbose=1
)


# ============================================================
# COMBINE HISTORY
# ============================================================

train_accuracy = (
    history1.history["accuracy"]
    +
    history2.history["accuracy"]
)


validation_accuracy = (
    history1.history["val_accuracy"]
    +
    history2.history["val_accuracy"]
)


train_loss = (
    history1.history["loss"]
    +
    history2.history["loss"]
)


validation_loss = (
    history1.history["val_loss"]
    +
    history2.history["val_loss"]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING BEST MODEL")
print("=" * 75)


best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print()
print("=" * 75)
print("VALIDATION EVALUATION")
print("=" * 75)


validation_loss_value, validation_accuracy_value = (
    best_model.evaluate(
        validation_generator,
        verbose=1
    )
)


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)


test_loss_value, test_accuracy_value = (
    best_model.evaluate(
        test_generator,
        verbose=1
    )
)


# ============================================================
# SAVE ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    validation_accuracy,
    label="Validation Accuracy"
)

plt.title(
    "Indian Food Classification - Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    ACCURACY_GRAPH,
    dpi=150
)

plt.close()


# ============================================================
# SAVE LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    train_loss,
    label="Training Loss"
)

plt.plot(
    validation_loss,
    label="Validation Loss"
)

plt.title(
    "Indian Food Classification - Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    LOSS_GRAPH,
    dpi=150
)

plt.close()


# ============================================================
# BIRYANI TEST PREDICTION CHECK
# ============================================================

print()
print("=" * 75)
print("BIRYANI MODEL CHECK")
print("=" * 75)


test_predictions = best_model.predict(
    test_generator,
    verbose=1
)


test_predicted_labels = np.argmax(
    test_predictions,
    axis=1
)


true_test_labels = (
    test_generator.classes
)


biryani_correct = 0
biryani_total = 0


if "biryani" in class_indices:

    biryani_index = class_indices[
        "biryani"
    ]

    biryani_positions = np.where(
        true_test_labels
        ==
        biryani_index
    )[0]


    biryani_total = len(
        biryani_positions
    )


    for position in biryani_positions:

        if (
            test_predicted_labels[
                position
            ]
            ==
            biryani_index
        ):

            biryani_correct += 1


biryani_accuracy = (

    biryani_correct
    /
    biryani_total

    if biryani_total > 0

    else 0.0
)


print(
    "Biryani test images:",
    biryani_total
)

print(
    "Biryani correct:",
    biryani_correct
)

print(
    f"Biryani accuracy: "
    f"{biryani_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

METRICS_PATH = os.path.join(
    ASSETS_DIR,
    "training_final_metrics.txt"
)


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "INDIAN FOOD IMAGE CLASSIFICATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Model               : EfficientNetB0\n"
    )

    f.write(
        f"Number of classes   : {NUM_CLASSES}\n"
    )

    f.write(
        f"Training images     : "
        f"{train_generator.samples}\n"
    )

    f.write(
        f"Validation images   : "
        f"{validation_generator.samples}\n"
    )

    f.write(
        f"Test images         : "
        f"{test_generator.samples}\n\n"
    )

    f.write(
        f"Validation accuracy : "
        f"{validation_accuracy_value * 100:.2f}%\n"
    )

    f.write(
        f"Test accuracy       : "
        f"{test_accuracy_value * 100:.2f}%\n"
    )

    f.write(
        f"Test loss           : "
        f"{test_loss_value:.4f}\n\n"
    )

    f.write(
        f"Biryani test images: "
        f"{biryani_total}\n"
    )

    f.write(
        f"Biryani correct     : "
        f"{biryani_correct}\n"
    )

    f.write(
        f"Biryani accuracy    : "
        f"{biryani_accuracy * 100:.2f}%\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 75)

print()
print(
    f"Validation Accuracy : "
    f"{validation_accuracy_value * 100:.2f}%"
)

print(
    f"Test Accuracy       : "
    f"{test_accuracy_value * 100:.2f}%"
)

print(
    f"Biryani Accuracy    : "
    f"{biryani_accuracy * 100:.2f}%"
)

print()
print(
    "Best model saved to:"
)

print(
    MODEL_PATH
)

print()
print(
    "Class names saved to:"
)

print(
    CLASS_NAMES_PATH
)

print()
print(
    "Accuracy graph:"
)

print(
    ACCURACY_GRAPH
)

print()
print(
    "Loss graph:"
)

print(
    LOSS_GRAPH
)

print()
print(
    "Metrics:"
)

print(
    METRICS_PATH
)

print()
print("=" * 75)
print("NEXT STEPS")
print("=" * 75)

print()
print("1. python evaluate.py")
print("2. python predict.py")
print("3. streamlit run app.py")

print()
print("=" * 75)