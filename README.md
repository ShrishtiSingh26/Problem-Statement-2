# Problem-Statement-2
# Iris Species Classification with K-Nearest Neighbors (KNN)

## Project Overview

This project uses the K-Nearest Neighbors (KNN) algorithm to classify iris flowers into one of three species based on four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The objective is to build a classification model that can predict the species of iris flowers using these features and evaluate its performance.



## Installation

### Requirements
- Python 3.7+
- Libraries: 
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install dependencies using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset
The dataset used is the popular Iris dataset, which is available in `sklearn.datasets`. It contains 150 samples, with 50 samples each for three species of iris flowers: Setosa, Versicolor, and Virginica.

## Solution Approach
Load the Dataset: The dataset is loaded using `load_iris` from `sklearn.datasets`.
Split the Dataset: Split the data into training and testing sets using `train_test_split`.
Train the Model: Train the K-Nearest Neighbors (KNN) classifier on the training data.
Evaluate the Model: Evaluate the classifier on the test set using metrics like accuracy, precision, recall, and F1 score.
Visualize the Results: Visualize the model performance with a confusion matrix and scatter plots of feature distributions.

## Evaluation Metrics
**Accuracy:** Measures overall correctness.
**Precision:** Measures the accuracy of positive predictions.
**Recall:** Measures the ability to find all positive instances.
**F1 Score:** Harmonic mean of precision and recall, useful for imbalanced datasets.

## Visualization
**Confusion Matrix:** Displays true vs. predicted labels, helping identify misclassifications.

![image](https://github.com/user-attachments/assets/c9ce91ac-edf2-41e3-86e7-6c954b6f644f)

**Feature Scatter Plot:** Provides a 2D visualization of sepal length and sepal width across species.

![image](https://github.com/user-attachments/assets/382c88f6-c363-4702-985d-bdec165a760d)


## Results
The model achieves high accuracy and performs well across all metrics for this balanced dataset.
