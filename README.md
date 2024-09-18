# Heart Attack Prediction Project

## Introduction
This project aims to develop a machine learning model to predict the likelihood of heart attacks using patient data. The goal is to leverage classification algorithms to identify key factors contributing to heart attacks and build a predictive model to aid in early diagnosis.

## Dataset Overview
The dataset contains 303 records with the following columns:
- **age**: Age of the patient
- **sex**: Gender of the patient (encoded)
- **cp**: Chest pain type (encoded)
- **trtbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (encoded)
- **restecg**: Resting electrocardiographic results (encoded)
- **thalachh**: Maximum heart rate achieved
- **exng**: Exercise induced angina (encoded)
- **oldpeak**: Depression induced by exercise relative to rest
- **slp**: Slope of the peak exercise ST segment (encoded)
- **caa**: Number of major vessels colored by fluoroscopy (encoded)
- **thall**: Thalassemia (encoded)
- **output**: Presence or absence of heart disease (target variable)

**For more Infomation**[here](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/data)

## Importing Libraries
I start by importing essential Python libraries for data manipulation, preprocessing, model building, and evaluation, including Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

## Loading Data
The dataset is loaded into a Pandas DataFrame, which provides an initial view of the data structure and content.

## Data Cleaning
I perform initial data inspection to identify and address any missing or inconsistent values. The dataset is cleaned to ensure it is ready for preprocessing.

## Data Preprocessing
Categorical variables are one-hot encoded, and numerical features are scaled. This prepares the data for modeling by transforming it into a format suitable for machine learning algorithms.

## Modeling
Several classification models are evaluated, including:
- **Logistic Regression**: A logistic regression model is trained and tuned for optimal performance.
- **SVM (Support Vector Machine)**: An SVM model is developed and tuned to classify heart attack cases.
- **KNN (K-Nearest Neighbors)**: The KNN algorithm is applied and optimized.
- **Decision Tree**: A decision tree model is created and fine-tuned.
- **Random Forest**: A random forest model is trained to handle complex feature interactions.
- **XGBoost**: An XGBoost model is utilized for its powerful gradient boosting capabilities.

## AUC for Models
The Area Under the Curve (AUC) is calculated for each model to assess its performance. AUC scores provide a measure of the model's ability to distinguish between classes.

## Logistic Regression Model with K-Fold
The Logistic Regression model is further evaluated using K-Fold cross-validation to ensure its robustness and generalizability. The best parameters and the optimal number of folds are identified.

## Data Profiling
A comprehensive analysis of the dataset's features and their distributions is conducted to gain insights into the data and its characteristics.

## Conclusion
The Logistic Regression model, after tuning and cross-validation, is identified as the best-performing model for heart attack prediction. This project demonstrates the effectiveness of various classification algorithms and provides a reliable model for predicting heart attack risk based on patient data.

![Importance](https://cityupload.io/2024/09/importance.png)
