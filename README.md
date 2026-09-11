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
```

## Activation Functions

The model uses:

* ReLU in the intermediate dense layer
* Softmax in the final classification layer

The Softmax layer produces a probability distribution across the three classes.

## 🔄 Transfer Learning

The project uses EfficientNetB0 pretrained on ImageNet.

The training process consists of two major stages:

### **Stage 1 — Feature Extraction**

The pretrained EfficientNetB0 convolutional base is frozen while the newly added classification layers are trained.

### **Stage 2 — Fine-Tuning**

The later portion of the pretrained network is unfrozen and fine-tuned using a much smaller learning rate.

This allows the model to adapt pretrained visual features to brain CT images while reducing the computational cost of training a CNN from scratch.

## 🧪 Data Preprocessing

The following preprocessing steps are implemented:

- Image loading
- RGB conversion
- Image resizing to 224 × 224
- Model-specific preprocessing
- Dataset splitting
- Stratified sampling

### Data Augmentation

Training images are augmented using:

- Rotation
- Width shifting
- Height shifting
- Zoom
- Shearing
- Horizontal flipping
- Nearest-neighbor filling

Data augmentation is applied only to the training set.

## 🔬 Feature Engineering

In addition to deep-learning features, several handcrafted numerical image features are extracted for exploratory analysis.

Examples include:

- Mean intensity
- Standard deviation of intensity
- Minimum intensity
- Maximum intensity
- Median intensity
- 25th percentile
- 75th percentile
- Dark pixel ratio
- Bright pixel ratio
- Edge strength
- Horizontal variance
- Vertical variance

These features are used for data analysis, scaling, feature selection and clustering.

## 📈 Exploratory Data Analysis

The project includes several visualization techniques.

### Class Distribution

A bar chart is used to examine the number of images in each class.

### Histograms

Histograms are generated for important engineered image features.

### Box Plots

Box plots are used to compare numerical feature distributions between classes.

### Scatter Plots

Scatter plots are used to examine relationships between engineered features.

### Regression Plots

Regression plots are used to examine relationships between numerical image features.

### Correlation Heatmap

A correlation heatmap is used to analyze relationships between engineered features.

### Pairplot

Pairplots provide a visual overview of relationships between multiple numerical features.

## ⚙️ Feature Scaling

The engineered numerical features are standardized using StandardScaler.

This transforms the features so that they have comparable scales.

## 🎯 Feature Selection

The project uses SelectKBest with ANOVA F-test to identify the most informative engineered features.

Feature-selection scores are also visualized using a bar chart.

## 🧠 Deep Learning

The primary deep-learning model is based on:

EfficientNetB0 + Transfer Learning

The project also provides support for other pretrained CNN architectures:

- MobileNetV2
- EfficientNetB0
- ResNet50
- VGG16
- DenseNet121

EfficientNetB0 is used as the default model because it provides a strong balance between model size, computational requirements and classification performance.

## ⏱️ Early Stopping

Early stopping is used during training to help prevent overfitting.

The model monitors validation loss and restores the best-performing model weights.

## 📊 Model Evaluation

The trained model is evaluated using an independent test set.

The evaluation includes:

- Test loss
- Test accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Normalized confusion matrix
- ROC curves
- AUC
- Prediction confidence
- Error analysis

## 📉 Training Curves

Training and validation curves are generated for:

### Accuracy
Training Accuracy vs. Validation Accuracy
### Loss
Training Loss vs. Validation Loss


These plots help analyze:

- Model convergence
- Overfitting
- Underfitting
- Training stability

## 🔲 Confusion Matrix

A confusion matrix is generated to visualize classification performance across the three classes.

Example structure:
```
                    Predicted
                B      I      N
             ┌──────┬──────┬──────┐
Actual   B   │      │      │      │
         I   │      │      │      │
         N   │      │      │      │
             └──────┴──────┴──────┘
```
Both regular and normalized confusion matrices are generated.

## 📈 ROC Curve and AUC

A multi-class ROC curve is generated using the One-vs-Rest (OvR) approach.

The project calculates:

- ROC curve for each class
- AUC for each class
- Macro-average ROC-AUC

