import os
import json
import hashlib
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FoodVision AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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

IMG_SIZE = (224, 224)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   MAIN APPLICATION
   ========================================================== */

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================================
   GLOBAL TEXT
   ========================================================== */

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #111827 !important;
}

[data-testid="stMain"] p {
    color: #374151;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero-box {
    background-color: #111827;
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 28px;
}

.hero-title {
    color: white !important;
    font-size: 42px;
    font-weight: 800;
    margin: 0;
}

.hero-subtitle {
    color: #d1d5db !important;
    font-size: 17px;
    margin-top: 8px;
}

.hero-tech {
    color: #c7d2fe !important;
    font-size: 14px;
    margin-top: 18px;
    font-weight: 600;
}


/* ==========================================================
   SECTION TITLE
   ========================================================== */

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #111827 !important;
    margin-top: 15px;
    margin-bottom: 15px;
}


/* ==========================================================
   INFORMATION CARDS
   ========================================================== */

.info-card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px;
    min-height: 105px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
}

.info-label {
    color: #6b7280 !important;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.info-value {
    color: #111827 !important;
    font-size: 21px;
    font-weight: 800;
    margin-top: 8px;
}


/* ==========================================================
   RESULT CARD
   ========================================================== */

.result-card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.05);
}

.result-label {
    color: #6b7280 !important;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-food {
    color: #111827 !important;
    font-size: 32px;
    font-weight: 800;
    margin-top: 5px;
}


/* ==========================================================
   TOP PREDICTION CARD
   ========================================================== */

.top-card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px;
    min-height: 150px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
}

.top-rank {
    color: #6b7280 !important;
    font-size: 12px;
    font-weight: 700;
}

.top-food {
    color: #111827 !important;
    font-size: 17px;
    font-weight: 750;
    margin-top: 8px;
}

.top-score {
    color: #111827 !important;
    font-size: 22px;
    font-weight: 800;
    margin-top: 7px;
}


/* ==========================================================
   HOW IT WORKS
   ========================================================== */

.step-card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px;
    min-height: 145px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
}

.step-number {
    color: #2563eb !important;
    font-size: 13px;
    font-weight: 800;
}

.step-title {
    color: #111827 !important;
    font-size: 18px;
    font-weight: 800;
    margin-top: 7px;
}

.step-text {
    color: #6b7280 !important;
    font-size: 14px;
    line-height: 1.5;
    margin-top: 6px;
}


/* ==========================================================
   FILE UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 15px;
}

[data-testid="stFileUploader"] section {
    background-color: white !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 15px !important;
}

[data-testid="stFileUploader"] section * {
    color: #111827 !important;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
}


/* ==========================================================
   METRICS
   ========================================================== */

[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: white !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #374151 !important;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;
    color: #6b7280 !important;
    font-size: 13px;
    padding-top: 20px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not os.path.exists(CLASS_NAMES_PATH):

        raise FileNotFoundError(
            f"Class names file not found:\n"
            f"{CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# DISPLAY NAME
# ============================================================

def display_name(name):

    return (
        str(name)
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


# ============================================================
# NORMALIZE NAME
# ============================================================

def normalize_name(name):

    return (
        str(name)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# FOOD INFORMATION
# ============================================================

FOOD_INFO = {

    "biryani":
        "Biryani is a rice-based dish prepared "
        "with aromatic spices and commonly served "
        "with vegetables, chicken, mutton or other ingredients.",

    "daal_puri":
        "Daal Puri is a flatbread prepared with "
        "seasoned lentils.",

    "dosa":
        "Dosa is a fermented rice and lentil crepe.",

    "idli":
        "Idli is a steamed food prepared from "
        "fermented rice and lentil batter.",

    "samosa":
        "Samosa is a crispy pastry commonly filled "
        "with spiced potato and peas.",

    "pav_bhaji":
        "Pav Bhaji is a spiced vegetable preparation "
        "served with bread rolls.",

    "poha":
        "Poha is a popular food prepared from "
        "flattened rice.",

    "puri":
        "Puri is a deep-fried bread commonly made "
        "from wheat flour.",

    "gulab_jamun":
        "Gulab Jamun is a sweet commonly served "
        "with sugar syrup."
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_food(model, image):

    # Convert image
    image = image.convert("RGB")

    # Resize
    image = image.resize(
        IMG_SIZE
    )

    # Convert to NumPy
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # ========================================================
    # IMPORTANT
    #
    # Current model = EfficientNetB0
    #
    # DO NOT use MobileNetV2 preprocess_input().
    #
    # EfficientNetB0 handles preprocessing internally.
    # ========================================================

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # ========================================================
    # PROBABILITY CHECK
    # ========================================================

    prediction_sum = np.sum(
        predictions
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

    # Remove invalid values
    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Normalize
    total = np.sum(
        predictions
    )

    if total > 0:

        predictions = (
            predictions / total
        )

    # Best prediction
    predicted_index = int(
        np.argmax(
            predictions
        )
    )

    confidence = float(
        predictions[
            predicted_index
        ]
    )

    return (
        predicted_index,
        confidence,
        predictions
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🍽️ FoodVision AI")

    st.caption(
        "Deep Learning Food Classification"
    )

    st.divider()

    st.subheader(
        "🤖 AI Model"
    )

    st.write(
        "EfficientNetB0"
    )

    st.write(
        "Transfer Learning"
    )

    st.divider()

    st.subheader(
        "📚 Dataset"
    )

    st.write(
        "80 Food Classes"
    )

    st.write(
        "224 × 224 Input Images"
    )

    st.divider()

    st.subheader(
        "🛠️ Technologies"
    )

    st.write(
        "Python"
    )

    st.write(
        "TensorFlow / Keras"
    )

    st.write(
        "Streamlit"
    )

    st.divider()

    st.subheader(
        "🎯 Project Goal"
    )

    st.write(
        "Automatically classify food images "
        "using a deep learning model."
    )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
<div class="hero-box">
    <div class="hero-title">
        🍽️ FoodVision AI
    </div>
    <div class="hero-subtitle">
        Intelligent Food Image Classification
        Using Deep Learning
    </div>
    <div class="hero-tech">
        ⚡ EfficientNetB0 &nbsp; • &nbsp;
        🧠 TensorFlow &nbsp; • &nbsp;
        🚀 Streamlit
    </div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

    class_names = load_class_names()

except Exception as error:

    st.error(
        "❌ Unable to load the AI model."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# MODEL / CLASS VALIDATION
# ============================================================

model_output_classes = (
    model.output_shape[-1]
)

if model_output_classes != len(
    class_names
):

    st.error(
        "❌ Model and class names do not match."
    )

    st.write(
        f"Model output classes: "
        f"{model_output_classes}"
    )

    st.write(
        f"Class names: "
        f"{len(class_names)}"
    )

    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

st.success(
    f"✅ AI model ready — "
    f"{len(class_names)} food classes available."
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Model Overview'
    '</div>',
    unsafe_allow_html=True
)

overview1, overview2, overview3, overview4 = st.columns(
    4
)


with overview1:

    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">AI MODEL</div>'
        '<div class="info-value">EfficientNetB0</div>'
        '</div>',
        unsafe_allow_html=True
    )


with overview2:

    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">FOOD CLASSES</div>'
        f'<div class="info-value">{len(class_names)}</div>'
        '</div>',
        unsafe_allow_html=True
    )


with overview3:

    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">IMAGE SIZE</div>'
        '<div class="info-value">224 × 224</div>'
        '</div>',
        unsafe_allow_html=True
    )


with overview4:

    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">FRAMEWORK</div>'
        '<div class="info-value">TensorFlow</div>'
        '</div>',
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# UPLOAD + PREDICTION
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT: UPLOAD
# ============================================================

with left_column:

    st.markdown(
        '<div class="section-title">'
        '📷 Upload Food Image'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a clear food image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is None:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 📤 Ready to Analyze"
            )

            st.write(
                "Upload a food image to start "
                "the classification process."
            )

            st.info(
                "Supported: JPG, JPEG, PNG, WEBP"
            )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"Original image: "
            f"{image.width} × {image.height} pixels"
        )


# ============================================================
# RIGHT: PREDICTION
# ============================================================

with right_column:

    st.markdown(
        '<div class="section-title">'
        '🔍 AI Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is None:

        with st.container(
            border=True
        ):

            st.markdown(
                "### ⏳ Waiting for Image"
            )

            st.write(
                "Upload an image on the left "
                "to begin prediction."
            )

    else:

        # ----------------------------------------------------
        # FILE HASH
        # ----------------------------------------------------

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()

        # ----------------------------------------------------
        # RESET WHEN IMAGE CHANGES
        # ----------------------------------------------------

        if (
            st.session_state.get(
                "last_file_hash"
            )
            != file_hash
        ):

            st.session_state[
                "last_file_hash"
            ] = file_hash

            st.session_state.pop(
                "predicted_food",
                None
            )

            st.session_state.pop(
                "confidence",
                None
            )

            st.session_state.pop(
                "predictions",
                None
            )

        # ----------------------------------------------------
        # PREDICT BUTTON
        # ----------------------------------------------------

        predict_clicked = st.button(
            "🔎 Predict Food",
            type="primary",
            use_container_width=True
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        if predict_clicked:

            with st.spinner(
                "AI is analyzing the image..."
            ):

                (
                    predicted_index,
                    confidence,
                    predictions
                ) = predict_food(
                    model,
                    image
                )

            if (
                predicted_index
                >=
                len(class_names)
            ):

                st.error(
                    "Model output does not "
                    "match the class names."
                )

                st.stop()

            predicted_food = (
                class_names[
                    predicted_index
                ]
            )

            confidence_percent = (
                confidence * 100
            )

            st.session_state[
                "predicted_food"
            ] = predicted_food

            st.session_state[
                "confidence"
            ] = confidence_percent

            st.session_state[
                "predictions"
            ] = predictions.tolist()

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        if (
            "predicted_food"
            in st.session_state
        ):

            predicted_food = (
                st.session_state[
                    "predicted_food"
                ]
            )

            confidence_percent = (
                st.session_state[
                    "confidence"
                ]
            )

            predictions = np.array(
                st.session_state[
                    "predictions"
                ]
            )

            food_name = display_name(
                predicted_food
            )

            st.markdown(
                '<div class="result-card">'
                '<div class="result-label">'
                'PREDICTED FOOD'
                '</div>'
                f'<div class="result-food">'
                f'🍽️ {food_name}'
                f'</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence_percent:.2f}%"
            )

            st.progress(
                min(
                    max(
                        confidence,
                        0.0
                    ),
                    1.0
                )
            )

            if confidence_percent >= 75:

                st.success(
                    "✅ High confidence prediction"
                )

            elif confidence_percent >= 50:

                st.warning(
                    "⚠️ Moderate confidence prediction"
                )

            else:

                st.warning(
                    "⚠️ Low confidence prediction"
                )

                st.info(
                    "Try a clearer image where "
                    "the food occupies most of "
                    "the image."
                )

            # ------------------------------------------------
            # FOOD INFORMATION
            # ------------------------------------------------

            normalized_food = normalize_name(
                predicted_food
            )

            if normalized_food in FOOD_INFO:

                st.subheader(
                    "🍴 About This Food"
                )

                st.info(
                    FOOD_INFO[
                        normalized_food
                    ]
                )


# ============================================================
# TOP 5 PREDICTIONS
# ============================================================

if (
    "predicted_food"
    in st.session_state
):

    predictions = np.array(
        st.session_state[
            "predictions"
        ]
    )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📊 Top 5 Predictions'
        '</div>',
        unsafe_allow_html=True
    )

    top_n = min(
        5,
        len(predictions),
        len(class_names)
    )

    top_indices = np.argsort(
        predictions
    )[-top_n:][::-1]

    top_columns = st.columns(
        top_n,
        gap="medium"
    )

    for rank, index in enumerate(
        top_indices
    ):

        index = int(
            index
        )

        name = display_name(
            class_names[index]
        )

        score = (
            float(
                predictions[index]
            ) * 100
        )

        with top_columns[rank]:

            st.markdown(
                '<div class="top-card">'
                f'<div class="top-rank">'
                f'#{rank + 1}'
                f'</div>'
                f'<div class="top-food">'
                f'{name}'
                f'</div>'
                f'<div class="top-score">'
                f'{score:.2f}%'
                f'</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    max(
                        float(
                            predictions[index]
                        ),
                        0.0
                    ),
                    1.0
                )
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '⚙️ How FoodVision AI Works'
    '</div>',
    unsafe_allow_html=True
)

step1, step2, step3, step4 = st.columns(
    4
)


with step1:

    st.markdown(
        '<div class="step-card">'
        '<div class="step-number">STEP 01</div>'
        '<div class="step-title">📤 Upload</div>'
        '<div class="step-text">'
        'Upload a clear food image using the application.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with step2:

    st.markdown(
        '<div class="step-card">'
        '<div class="step-number">STEP 02</div>'
        '<div class="step-title">🖼️ Preprocess</div>'
        '<div class="step-text">'
        'The image is converted to RGB and resized to '
        '224 × 224 pixels.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with step3:

    st.markdown(
        '<div class="step-card">'
        '<div class="step-number">STEP 03</div>'
        '<div class="step-title">🧠 Classify</div>'
        '<div class="step-text">'
        'EfficientNetB0 analyzes the image and generates '
        'class probabilities.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with step4:

    st.markdown(
        '<div class="step-card">'
        '<div class="step-number">STEP 04</div>'
        '<div class="step-title">🎯 Result</div>'
        '<div class="step-text">'
        'The application displays the predicted food and '
        'top alternative predictions.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# TECHNOLOGY SECTION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '🛠️ Technology Stack'
    '</div>',
    unsafe_allow_html=True
)

tech1, tech2, tech3, tech4 = st.columns(
    4
)


with tech1:

    st.info(
        "🐍 **Python**\n\n"
        "Core programming language"
    )


with tech2:

    st.info(
        "🧠 **TensorFlow / Keras**\n\n"
        "Deep learning framework"
    )


with tech3:

    st.info(
        "⚡ **EfficientNetB0**\n\n"
        "Transfer learning model"
    )


with tech4:

    st.info(
        "🚀 **Streamlit**\n\n"
        "Interactive web interface"
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '📌 Project Information'
    '</div>',
    unsafe_allow_html=True
)

info_left, info_right = st.columns(
    2
)


with info_left:

    st.write(
        "**Project:** Food Image Classification"
    )

    st.write(
        "**Model:** EfficientNetB0"
    )

    st.write(
        "**Learning Approach:** Transfer Learning"
    )


with info_right:

    st.write(
        f"**Classes:** {len(class_names)}"
    )

    st.write(
        "**Input Resolution:** 224 × 224"
    )

    st.write(
        "**Interface:** Streamlit Web Application"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🍽️ FoodVision AI &nbsp;|&nbsp; '
    'Food Image Classification &nbsp;|&nbsp; '
    'EfficientNetB0 + TensorFlow + Streamlit'
    '</div>',
    unsafe_allow_html=True
)