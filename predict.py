import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "model/food_classifier.keras"
CLASS_NAMES_PATH = "model/class_names.json"

IMG_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

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


print(
    f"Number of classes: {len(class_names)}"
)


# ============================================================
# VERIFY MODEL / CLASS COUNT
# ============================================================

output_classes = model.output_shape[-1]

print(
    f"Model output classes: {output_classes}"
)

if output_classes != len(class_names):

    raise ValueError(
        f"Mismatch detected!\n"
        f"Model classes: {output_classes}\n"
        f"Class names: {len(class_names)}"
    )


# ============================================================
# VERIFY BIRYANI CLASS
# ============================================================

if "biryani" in class_names:

    biryani_index = class_names.index(
        "biryani"
    )

    print(
        f"Biryani class index: {biryani_index}"
    )

else:

    biryani_index = None

    print(
        "WARNING: Biryani is not in class_names.json"
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_food(
    img_path,
    top_k=10
):

    if not os.path.exists(img_path):

        raise FileNotFoundError(
            f"Image not found: {img_path}"
        )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )


    # --------------------------------------------------------
    # CONVERT IMAGE TO ARRAY
    # --------------------------------------------------------

    img_array = image.img_to_array(
        img
    )


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------------
    # MOBILE NET V2 PREPROCESSING
    # --------------------------------------------------------

    img_array = preprocess_input(
        img_array
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]


    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if len(predictions) != len(
        class_names
    ):

        raise ValueError(
            "Prediction output does not "
            "match class_names.json"
        )


    # --------------------------------------------------------
    # HANDLE LOGITS IF NECESSARY
    # --------------------------------------------------------

    prediction_sum = float(
        np.sum(predictions)
    )


    if (
        np.any(predictions < 0)
        or
        np.any(predictions > 1)
        or
        not np.isclose(
            prediction_sum,
            1.0,
            atol=0.01
        )
    ):

        predictions = tf.nn.softmax(
            predictions
        ).numpy()


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    total = float(
        np.sum(predictions)
    )

    if total > 0:

        predictions = (
            predictions / total
        )


    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[::-1][:top_k]


    results = []


    for index in top_indices:

        index = int(index)

        results.append(
            {
                "food":
                    class_names[index],

                "confidence":
                    float(
                        predictions[index] * 100
                    ),

                "index":
                    index
            }
        )


    return (
        results,
        predictions
    )


# ============================================================
# TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("INDIAN FOOD IMAGE CLASSIFIER")
    print("=" * 70)


    img_path = input(
        "\nEnter image path: "
    ).strip().strip('"')


    try:

        (
            results,
            predictions
        ) = predict_food(
            img_path,
            top_k=10
        )


        # ====================================================
        # TOP 10
        # ====================================================

        print("\nTop 10 Predictions")
        print("-" * 70)


        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{i:2}. "
                f"{result['food']:<25} "
                f"{result['confidence']:>7.2f}% "
                f"(index {result['index']})"
            )


        print("-" * 70)


        # ====================================================
        # BEST PREDICTION
        # ====================================================

        best = results[0]


        print(
            f"\nPredicted Food : "
            f"{best['food']}"
        )


        print(
            f"Confidence     : "
            f"{best['confidence']:.2f}%"
        )


        print(
            f"Class Index    : "
            f"{best['index']}"
        )


        # ====================================================
        # BIRYANI SPECIFIC CHECK
        # ====================================================

        if biryani_index is not None:

            biryani_probability = (
                float(
                    predictions[
                        biryani_index
                    ]
                ) * 100
            )


            print("\n")
            print(
                "=" * 70
            )

            print(
                "BIRYANI CHECK"
            )

            print(
                "=" * 70
            )


            print(
                f"Biryani index : "
                f"{biryani_index}"
            )


            print(
                f"Biryani score : "
                f"{biryani_probability:.2f}%"
            )


            if best["food"] == "biryani":

                print(
                    "\n✅ The model predicted BIRYANI."
                )

            else:

                print(
                    "\n❌ The model did NOT predict Biryani."
                )


            print(
                "=" * 70
            )


    except Exception as e:

        print("\nPrediction Error:")
        print(e)