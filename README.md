# Brain-Stroke-Detection-from-CT-scan-images-using-Transfer-Learning
An end-to-end deep learning pipeline built for automated multi-class classification of brain computed tomography (CT) scans into normal, ischemia, and bleeding categories using transfer learning and state-of-the-art convolutional neural networks.
- 🩸 **Bleeding**
- 🧠 **Ischemia**
- ✅ **Normal**

The project includes complete data preprocessing, exploratory data analysis, feature engineering, feature selection, clustering, CNN-based transfer learning, model evaluation, error analysis, and a Streamlit web application for making predictions on new CT images.

## 📌 Project Overview

Brain stroke is a serious medical condition that requires rapid diagnosis. Medical imaging, particularly CT scans, can provide important information for identifying abnormalities associated with stroke.

This project explores the use of **deep learning and transfer learning** to automatically classify brain CT images.

Instead of training a large convolutional neural network entirely from scratch, a pretrained **EfficientNetB0** model trained on ImageNet is used as the feature extraction backbone. The model is then adapted and fine-tuned for the brain CT classification task.

The project also includes several traditional machine-learning and data-analysis techniques to provide a complete machine-learning workflow.

## 🎯 Objectives

The main objectives of this project are:

- Develop a brain CT image classification model.
- Apply transfer learning using a pretrained CNN.
- Use **EfficientNetB0** as the primary pretrained model.
- Perform image preprocessing and data augmentation.
- Perform exploratory data analysis (EDA).
- Engineer numerical image features.
- Apply feature scaling and feature selection.
- Explore the engineered features using PCA.
- Apply K-Means clustering.
- Train and fine-tune a CNN-based classifier.
- Use Early Stopping to reduce overfitting.
- Evaluate the model on an independent test set.
- Visualize model performance and prediction errors.
- Build a web-based prediction interface using Streamlit.

# 📊 Dataset

The project uses the **Brain Stroke CT Dataset** available on Kaggle.

### Dataset Source

[Kaggle - Brain Stroke CT Dataset](https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset)

The dataset contains brain CT images belonging to three classes:

| Class | Description |
|---|---|
| **Bleeding** | CT images associated with intracranial bleeding |
| **Ischemia** | CT images associated with ischemic stroke |
| **Normal** | CT images without the target stroke abnormalities |

The dataset is divided into training, validation, and testing subsets using a **stratified split**.

# 🏗️ Model Architecture

The main model uses **EfficientNetB0** with transfer learning.

### Architecture

```text
Input Brain CT Image
        │
        ▼
Image Resizing (224 × 224)
        │
        ▼
Preprocessing
        │
        ▼
Pretrained EfficientNetB0
        │
        ▼
Global Average Pooling
        │
        ▼
Dense Layer (256 neurons)
        │
        ▼
Batch Normalization
        │
        ▼
Dropout (0.40)
        │
        ▼
Dense Layer (3 neurons)
        │
        ▼
Softmax
        │
        ▼
┌─────────────┬─────────────┬─────────────┐
│  Bleeding   │  Ischemia   │   Normal    │
└─────────────┴─────────────┴─────────────┘
