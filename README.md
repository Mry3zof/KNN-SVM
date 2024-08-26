# Medical Transcription Classification using KNN and SVM

## 1. Problem and Aim

The problem is to classify medical transcriptions into different categories using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) Algorithms. The goal is to develop models that accurately predict the classification of medical transcriptions based on the given features. This task is crucial for creating accurate medical records.

## 1.2 Objectives

- Implement KNN and SVM models to classify medical transcription data.
- Evaluate the performance of these models.
- Compare the results to determine which algorithm performs better on the dataset.

Expected outcomes:
- A trained KNN model with the best hyperparameters for classification.
- A trained SVM model with optimal parameters.
- Performance metrics (accuracy) for both models to assess their effectiveness.

## 2. Data Preparation

### 2.1 Data Collection

- **Dataset**: The dataset used is from the Hugging Face `datasets` library.
- **Source**: The dataset is available at `DataFog/medical-transcription-instruct`.
- **Summary**:
  - **Number of Features**: Two features, `transcription` (text) and `sample_name` (label).
  - **Number of Samples**: Varies based on the dataset split.

## 3. Machine Learning Algorithms

### K-Nearest Neighbors (KNN)

KNN is a simple, instance-based learning algorithm where the class of a sample is determined by the majority vote of its nearest neighbors. 
- **Hyperparameters**:
  - Number of neighbors (`k`): Set to 3 in this project.
  - Distance metric: Default Euclidean distance.

### Support Vector Machine (SVM)

SVM is a supervised learning algorithm that finds the hyperplane that best separates different classes in the feature space.
- **Hyperparameters**:
  - Kernel: Linear kernel.
  - Regularization parameter (`C`): Set to 1.0.

## 4. Model Evaluation

### Metrics

- **Accuracy**: The proportion of correctly classified samples out of the total number of samples. It is used to evaluate the performance of both KNN and SVM models.

## References

- 1. HCIA-AI V3.5 Training matrerials
  2. https://stackoverflow.com/questions/49277926/python-tf-idf-algorithm
  3. https://huggingface.co/datasets/DataFog/medical-transcription-instruct
  4. https://youtube.com/playlist?list=PL6-3IRz2XF5U98PPtkc34sg7EEGC34WRs&feature=shared

## Summary

This project implements and evaluates K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) models for classifying medical transcriptions. The models are trained and tested on a dataset from Hugging Face, and their performance is assessed using accuracy as the primary metric. The results provide insights into which algorithm performs better for this specific classification task and demonstrate practical applications of these machine learning techniques in medical data analysis.

---
