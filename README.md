# Iris Species Classification using Perceptron & Deep Neural Networks (ANN)
An end-to-end machine learning and deep learning project classifying Iris flower species using both a classical Single-Layer Perceptron and a Multi-Layer Feedforward Neural Network (ANN) built with TensorFlow/Keras.
---
## 📌 Project Overview
The objective of this project is to explore and contrast the performance of a baseline linear classifier against a deep learning architecture on the multiclass Iris flower dataset.
- **Dataset**: Iris Dataset (150 samples, 4 features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`)
- **Target Classes**: `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` (50 samples each)
- **Task**: Multiclass Classification
---
## ⚙️ Workflow & Architecture
### 1. Data Preprocessing
- **Label Encoding**: Encoded target species labels into numerical integers ($0, 1, 2$) using `LabelEncoder`.
- **Stratified Train-Test Split**: Divided dataset into 80% training and 20% testing sets while preserving target class distributions (`stratify=y`).
- **Feature Scaling**: Standardized numerical features using `StandardScaler` to ensure zero mean and unit variance.
- **One-Hot Encoding**: Converted class labels into binary class matrices using Keras's `to_categorical` for neural network training.
### 2. Models Implemented
- **Baseline Classifier**: Scikit-Learn `Perceptron` (`max_iter=1000`, `random_state=42`)
- **Deep Learning Model**: Fully Connected Sequential Artificial Neural Network (ANN):
  - **Input Layer**: 4 features
  - **Dense Layer 1**: 16 neurons with `ReLU` activation
  - **Dense Layer 2**: 8 neurons with `ReLU` activation
  - **Output Layer**: 3 neurons with `Softmax` activation
  - **Compilation**: Adam optimizer, Categorical Crossentropy loss function, Accuracy metric
---
## 📊 Results & Performance

| Model | Test Accuracy | Notes |
| :--- | :--- | :--- |
| **Perceptron (Baseline)** | **~86.67%** | Precision: 0.83–0.90 across classes |
| **Neural Network (ANN)** | **~93.33%** | Trained for 100 epochs (batch size = 8) |

The multi-layer neural network effectively captures nonlinear boundaries between overlapping classes (`versicolor` and `virginica`), outperforming the single-layer perceptron.
---
## 🛠️ Tech Stack & Libraries
- **Language**: Python
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Deep Learning**: `tensorflow` / `keras`
---
## 🙏 Acknowledgements
A heartfelt thank you to **Sheryians AI School** and its mentors for providing structured, practical guidance in Deep Learning and Neural Networks. The clear explanations of foundational concepts—ranging from perceptrons to multi-layer architectures—made building and understanding this project possible.
