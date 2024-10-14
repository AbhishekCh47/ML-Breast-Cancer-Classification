
# Breast Cancer Classification - Machine Learning Project

## Project Overview

This project focuses on predicting the type of breast cancer—Malignant or Benign—based on characteristics of cells from a given dataset. The goal is to accurately classify the tumors using machine learning algorithms, specifically **K-Nearest Neighbors (KNN)** and **Logistic Regression (LR)**.

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Algorithms Used](#algorithms-used)
3. [Project Files](#project-files)
4. [Requirements](#requirements)
5. [Running the Project](#running-the-project)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)

---

## Problem Statement

The task is to predict whether breast cancer is **Malignant** or **Benign** based on the characteristics of diseased cells. The dataset used has the following class distribution:

- **Malignant Tumors**: 34.5%
- **Benign Tumors**: 65.5%

The aim is to classify the data effectively to predict whether the breast cancer is malignant or benign.

---

## Algorithms Used

### 1. K-Nearest Neighbors (KNN)
- **Description**: KNN is a simple, lazy learning algorithm used for both classification and regression. It works by finding the 'k' most similar records (neighbors) from the training data and making predictions based on those neighbors.
- **Steps**:
  - Calculate Euclidean Distance between points.
  - Identify the nearest neighbors.
  - Make predictions based on the majority class of the neighbors.

### 2. Logistic Regression (LR)
- **Description**: Logistic Regression is a supervised classification algorithm based on the concept of probabilities. Unlike linear regression, it models binary outputs (e.g., 0 or 1).
- **Details**: 
  - It uses a sigmoid function to map predicted values to probabilities.
  - Suitable for binary classification tasks like predicting whether a tumor is malignant or benign.

---

## Project Files

- `src.py`: This file contains the code implementation of the KNN and Logistic Regression algorithms, as well as the dataset loading, preprocessing, and result evaluation.
- `MACHINE LEARNING.pptx`: The presentation file provides an overview of the project, algorithms used, results, and conclusions.

---

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `sklearn`
  - `seaborn`

---

## Running the Project

1. Clone the repository or download the project files.
2. Ensure all required libraries are installed:
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn
   ```
3. Run the Python script `src.py`:
   ```bash
   python src.py
   ```

This will execute the KNN and Logistic Regression models and display the results, including accuracy scores, F1 scores, and performance graphs.

---

## Results

- **Performance Metrics**:
  - Accuracy: Both KNN and Logistic Regression achieved similar accuracy.
  - **F1 Score**: The weighted harmonic mean of the precision and recall was used to assess the accuracy of the models.
  
- **Execution Time**: KNN was observed to take longer to compute compared to Logistic Regression due to its lazy learning nature, but the overall accuracy for both algorithms was nearly the same.

---

## Conclusion

- Both KNN and Logistic Regression provide effective means for classifying breast cancer tumors.
- While KNN takes longer to compute, it can still be valuable in certain scenarios, though Logistic Regression offers a quicker solution with comparable accuracy.

---

## Future Work

- Explore more advanced algorithms like Support Vector Machines (SVM) or Neural Networks to improve classification accuracy.
- Perform feature engineering to enhance the input data quality.
- Investigate the model's performance under different dataset distributions and introduce cross-validation for better generalization.

---

