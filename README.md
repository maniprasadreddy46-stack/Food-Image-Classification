# 🍽️ Food Image Classification Using Deep Learning

A deep learning based food image classification system that identifies food items from uploaded images using **EfficientNetB0**, **TensorFlow/Keras**, and **Streamlit**.

## 📌 Project Overview

Food image classification is a computer vision application that automatically recognizes different food categories from images.

This project uses transfer learning with the EfficientNetB0 architecture to classify food images into **80 food classes**.

The trained model is integrated with a professional Streamlit web interface where users can upload a food image and receive the predicted food category along with its confidence score and top predictions.

## ✨ Features

- 📷 Upload food images
- 🤖 AI-based food classification
- ⚡ EfficientNetB0 transfer learning
- 🍽️ Support for 80 food classes
- 📊 Prediction confidence score
- 🏆 Top-5 predictions
- 📖 Food information
- 🌐 Interactive Streamlit interface
- 📱 Simple and user-friendly design

## 🧠 AI Model

**Model:** EfficientNetB0

**Approach:** Transfer Learning

**Input Size:** 224 × 224 pixels

**Output:** 80 food classes

The EfficientNetB0 model uses pretrained ImageNet features and is fine-tuned for food image classification.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Pillow
- Matplotlib
- Streamlit
- Git
- GitHub

## 📂 Project Structure

```text
Food-Image-Classification/
│
├── assets/
│   ├── training_accuracy_final.png
│   ├── training_loss_final.png
│   └── evaluation results
│
├── model/
│   ├── food_classifier.keras
│   └── class_names.json
│
├── app.py
├── train.py
├── train_final.py
├── evaluate.py
├── inference.py
├── predict.py
├── prepare_dataset.py
├── food_info.py
├── requirements.txt
├── .gitignore
└── README.md




Food Image
     ↓
Image Upload
     ↓
Image Preprocessing
     ↓
EfficientNetB0
     ↓
Feature Extraction
     ↓
Classification Layer
     ↓
Food Prediction
     ↓
Confidence Score
     ↓
Top-5 Predictions