import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 30

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

MODEL_DIR = "model"
ASSETS_DIR = "assets"

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
    "training_accuracy.png"
)

LOSS_GRAPH = os.path.join(
    ASSETS_DIR,
    "training_loss.png"
)

CSV_LOG = os.path.join(
    ASSETS_DIR,
    "training_history.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ============================================================
# GPU CONFIGURATION
# ============================================================

print("=" * 75)
print("INDIAN FOOD IMAGE CLASSIFICATION")
print("Improved MobileNetV2 Transfer Learning")
print("=" * 75)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(" -", gpu)
else:
    print("\nNo GPU detected.")
    print("Training will use CPU.")


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Validation folder not found: {VAL_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test folder not found: {TEST_DIR}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

print("\nCreating image generators...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    rotation_range=25,

    width_shift_range=0.15,
    height_shift_range=0.15,

    shear_range=0.15,

    zoom_range=0.20,

    horizontal_flip=True,

    brightness_range=[0.70, 1.30],

    fill_mode="nearest"
)


val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# ============================================================
# TRAIN GENERATOR
# ============================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True,

    seed=42
)


# ============================================================
# VALIDATION GENERATOR
# ============================================================

validation_generator = val_datagen.flow_from_directory(

    VAL_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# ============================================================
# TEST GENERATOR
# ============================================================

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

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

print("\n")
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

print("=" * 75)


# ============================================================
# CREATE CLASS NAME LIST
# ============================================================

class_indices = train_generator.class_indices

class_names = [None] * NUM_CLASSES

for class_name, index in class_indices.items():

    class_names[index] = class_name


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


print("\nClass names saved to:")
print(CLASS_NAMES_PATH)


# ============================================================
# DISPLAY CLASS INDICES
# ============================================================

print("\nChecking important classes...")

if "biryani" in class_indices:

    print(
        "Biryani class index:",
        class_indices["biryani"]
    )

else:

    print(
        "WARNING: Biryani class not found!"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

class_counts = {}

for class_name, class_index in class_indices.items():

    class_folder = os.path.join(
        TRAIN_DIR,
        class_name
    )

    count = 0

    if os.path.exists(class_folder):

        for filename in os.listdir(class_folder):

            filepath = os.path.join(
                class_folder,
                filename
            )

            if os.path.isfile(filepath):

                count += 1

    class_counts[class_name] = count

    print(
        f"{class_name:<30} : {count}"
    )

print("=" * 75)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

labels = []

for class_name, class_index in class_indices.items():

    count = class_counts[class_name]

    labels.extend(
        [class_index] * count
    )


labels = np.array(labels)

unique_classes = np.unique(labels)


weights = compute_class_weight(
    class_weight="balanced",

    classes=unique_classes,

    y=labels
)


class_weights = {

    int(class_index): float(weight)

    for class_index, weight
    in zip(unique_classes, weights)

}


print("\nClass weights calculated.")

print("\nSample class weights:")

for class_index in list(class_weights.keys())[:10]:

    print(
        f"{class_names[class_index]:<30}"
        f": {class_weights[class_index]:.4f}"
    )


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print("\n")
print("=" * 75)
print("LOADING MOBILENETV2")
print("=" * 75)


base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# ============================================================
# FREEZE BASE MODEL
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

x = BatchNormalization()(x)

x = Dropout(0.40)(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.30)(x)

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
        learning_rate=0.0003
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


print("\nModel created successfully.")


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


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

    patience=8,

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
# PHASE 1
# ============================================================

print("\n")
print("=" * 75)
print("PHASE 1 - TRANSFER LEARNING")
print("=" * 75)

print(
    f"Training for maximum {INITIAL_EPOCHS} epochs..."
)


history1 = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

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

print("\n")
print("=" * 75)
print("PHASE 2 - FINE TUNING MOBILENETV2")
print("=" * 75)


# Unfreeze MobileNetV2
base_model.trainable = True


# Freeze most early layers
# Train only the later layers
fine_tune_from = len(base_model.layers) - 60


for layer in base_model.layers[
    :fine_tune_from
]:

    layer.trainable = False


# Keep BatchNormalization layers frozen
for layer in base_model.layers:

    if isinstance(
        layer,
        BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# TRAINABLE LAYER INFORMATION
# ============================================================

trainable_count = 0
frozen_count = 0

for layer in model.layers:

    if layer.trainable:

        trainable_count += 1

    else:

        frozen_count += 1


print(
    "\nTrainable layers:",
    trainable_count
)

print(
    "Frozen layers   :",
    frozen_count
)


# ============================================================
# RECOMPILE FOR FINE TUNING
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.00002
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# FINE-TUNING TRAINING
# ============================================================

history2 = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

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


val_accuracy = (

    history1.history["val_accuracy"]

    +

    history2.history["val_accuracy"]

)


train_loss = (

    history1.history["loss"]

    +

    history2.history["loss"]

)


val_loss = (

    history1.history["val_loss"]

    +

    history2.history["val_loss"]

)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n")
print("=" * 75)
print("LOADING BEST MODEL")
print("=" * 75)


best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\nEvaluating validation dataset...")


val_loss_value, val_accuracy_value = (

    best_model.evaluate(

        validation_generator,

        verbose=1

    )

)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating test dataset...")


test_loss_value, test_accuracy_value = (

    best_model.evaluate(

        test_generator,

        verbose=1

    )

)


# ============================================================
# BEST ACCURACIES
# ============================================================

best_training_accuracy = max(
    train_accuracy
)

best_validation_accuracy = max(
    val_accuracy
)


# ============================================================
# TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
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

plt.grid(True)

plt.tight_layout()

plt.savefig(
    ACCURACY_GRAPH,
    dpi=150
)

plt.close()


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    train_loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
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

plt.grid(True)

plt.tight_layout()

plt.savefig(
    LOSS_GRAPH,
    dpi=150
)

plt.close()


# ============================================================
# SAVE CLASS NAMES AGAIN
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


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FINAL TRAINING RESULTS")
print("=" * 75)


print(
    f"Best Training Accuracy    : "
    f"{best_training_accuracy * 100:.2f}%"
)


print(
    f"Best Validation Accuracy  : "
    f"{best_validation_accuracy * 100:.2f}%"
)


print(
    f"Final Validation Accuracy : "
    f"{val_accuracy_value * 100:.2f}%"
)


print(
    f"Final Test Accuracy       : "
    f"{test_accuracy_value * 100:.2f}%"
)


print(
    f"Final Test Loss           : "
    f"{test_loss_value:.4f}"
)


print("=" * 75)


# ============================================================
# BIRYANI DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 75)
print("BIRYANI DATASET INFORMATION")
print("=" * 75)


if "biryani" in class_indices:

    biryani_index = class_indices[
        "biryani"
    ]

    biryani_train_path = os.path.join(
        TRAIN_DIR,
        "biryani"
    )

    biryani_val_path = os.path.join(
        VAL_DIR,
        "biryani"
    )

    biryani_test_path = os.path.join(
        TEST_DIR,
        "biryani"
    )


    biryani_train_count = len([
        f for f in os.listdir(
            biryani_train_path
        )
        if os.path.isfile(
            os.path.join(
                biryani_train_path,
                f
            )
        )
    ])


    biryani_val_count = len([
        f for f in os.listdir(
            biryani_val_path
        )
        if os.path.isfile(
            os.path.join(
                biryani_val_path,
                f
            )
        )
    ])


    biryani_test_count = len([
        f for f in os.listdir(
            biryani_test_path
        )
        if os.path.isfile(
            os.path.join(
                biryani_test_path,
                f
            )
        )
    ])


    print(
        "Biryani class index      :",
        biryani_index
    )

    print(
        "Biryani training images  :",
        biryani_train_count
    )

    print(
        "Biryani validation images:",
        biryani_val_count
    )

    print(
        "Biryani test images      :",
        biryani_test_count
    )

else:

    print(
        "Biryani class was not found!"
    )


# ============================================================
# FILE INFORMATION
# ============================================================

print("\n")
print("=" * 75)
print("OUTPUT FILES")
print("=" * 75)


print(
    "\nBest model:"
)

print(
    MODEL_PATH
)


print(
    "\nClass names:"
)

print(
    CLASS_NAMES_PATH
)


print(
    "\nAccuracy graph:"
)

print(
    ACCURACY_GRAPH
)


print(
    "\nLoss graph:"
)

print(
    LOSS_GRAPH
)


print(
    "\nTraining history:"
)

print(
    CSV_LOG
)


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 75)
print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
print("=" * 75)


print("\nNext steps:")

print(
    "1. python evaluate.py"
)

print(
    "2. python predict.py"
)

print(
    "3. streamlit run app.py"
)

print("\nTest the same Biryani image again.")

print("=" * 75)