import os
import sys
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "model/food_classifier.keras"
CLASS_NAMES_PATH = "model/class_names.json"

IMG_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("INDIAN FOOD IMAGE CLASSIFIER - MODEL TEST")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

print("Number of classes:", len(class_names))
print("Model output classes:", model.output_shape[-1])


if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        "ERROR: Model classes and class_names.json do not match!"
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_food(image_path, top_k=10):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    print("\nTesting image:")
    print(image_path)

    # Load image
    img = image.load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    # Convert image
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    img_array = preprocess_input(
        img_array
    )

    # Prediction
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # Make sure probabilities are valid
    if (
        np.any(predictions < 0)
        or
        np.any(predictions > 1)
        or
        not np.isclose(
            np.sum(predictions),
            1.0,
            atol=0.01
        )
    ):

        predictions = tf.nn.softmax(
            predictions
        ).numpy()

    predictions = np.nan_to_num(
        predictions
    )

    total = np.sum(predictions)

    if total > 0:
        predictions = predictions / total

    # Top predictions
    top_indices = np.argsort(
        predictions
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        index = int(index)

        results.append({
            "index": index,
            "food": class_names[index],
            "confidence": float(
                predictions[index] * 100
            )
        })

    return results, predictions


# ============================================================
# GET IMAGE PATH
# ============================================================

if len(sys.argv) > 1:

    image_path = sys.argv[1]

else:

    print("\nNo image path supplied.")

    image_path = input(
        "\nEnter image path: "
    ).strip().strip('"')


# ============================================================
# RUN PREDICTION
# ============================================================

try:

    results, predictions = predict_food(
        image_path,
        top_k=10
    )

    print("\n")
    print("=" * 70)
    print("TOP 10 PREDICTIONS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        food = result["food"].replace(
            "_",
            " "
        ).title()

        confidence = result["confidence"]

        print(
            f"{rank:2}. "
            f"{food:<30} "
            f"{confidence:>7.2f}%"
        )

    print("=" * 70)


    # ========================================================
    # BEST PREDICTION
    # ========================================================

    best = results[0]

    best_food = best["food"].replace(
        "_",
        " "
    ).title()

    print("\nPREDICTED FOOD")
    print("-" * 70)

    print(
        f"Food       : {best_food}"
    )

    print(
        f"Confidence : {best['confidence']:.2f}%"
    )

    print(
        f"Class Index: {best['index']}"
    )


    # ========================================================
    # BIRYANI CHECK
    # ========================================================

    if "biryani" in class_names:

        biryani_index = class_names.index(
            "biryani"
        )

        biryani_score = (
            float(
                predictions[biryani_index]
            ) * 100
        )

        print("\n")
        print("=" * 70)
        print("BIRYANI CHECK")
        print("=" * 70)

        print(
            f"Biryani index : {biryani_index}"
        )

        print(
            f"Biryani score : {biryani_score:.2f}%"
        )

        if best["food"] == "biryani":

            print(
                "\n✅ MODEL PREDICTED BIRYANI"
            )

        else:

            print(
                "\n❌ MODEL DID NOT PREDICT BIRYANI"
            )

        print("=" * 70)


except Exception as e:

    print("\n")
    print("=" * 70)
    print("PREDICTION ERROR")
    print("=" * 70)

    print(e)