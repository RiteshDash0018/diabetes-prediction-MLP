# Diabetes Prediction using Multilayer Perceptron

## Project Overview

This project presents a machine learning-based system for predicting diabetes using the Pima Indians Diabetes Dataset.

The main objective of the project is to develop and evaluate multiple machine learning classification models and to investigate the performance of a Multilayer Perceptron (MLP) for diabetes prediction.

The project follows a complete machine learning workflow, including:

- Dataset understanding
- Exploratory Data Analysis (EDA)
- Data cleaning
- Missing and invalid value handling
- Feature engineering
- Feature selection
- Train-test splitting
- Feature scaling
- Baseline model development
- Multilayer Perceptron development
- Regularization experiments
- Hyperparameter tuning
- Model evaluation
- Final model training
- Model saving
- Streamlit deployment
- Deployment testing

---

## Problem Statement

Diabetes prediction is a binary classification problem in which patient-related medical and demographic features are used to predict whether an individual is classified as diabetic or non-diabetic.

The project aims to develop a machine learning model that can learn patterns from patient data and generate a diabetes prediction.

Multiple classification algorithms are compared, with particular focus on developing, tuning, evaluating, and deploying a Multilayer Perceptron model.

---

## Objectives

The main objectives of this project are:

1. To understand and analyze the Pima Indians Diabetes Dataset.
2. To perform exploratory data analysis.
3. To identify invalid and physiologically implausible values.
4. To handle missing and invalid values appropriately.
5. To perform feature engineering.
6. To select suitable features for model development.
7. To split the dataset into training and testing sets.
8. To scale the input features.
9. To develop conventional machine learning classification models.
10. To develop a Multilayer Perceptron model.
11. To investigate different activation functions.
12. To experiment with regularization techniques.
13. To tune important MLP hyperparameters.
14. To compare the performance of different models.
15. To train and evaluate the final tuned MLP.
16. To deploy the final model using Streamlit.

---

# Dataset

## Dataset Used

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains medical and demographic information of patients and uses the `Outcome` variable as the target for diabetes classification.

## Dataset Size

- Number of observations: **768**
- Number of columns: **9**
- Input features: **8**
- Target variable: **Outcome**

## Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of the patient |
| Outcome | Diabetes classification |

## Target Variable

The `Outcome` variable represents the target class:

- `0` = Non-Diabetic
- `1` = Diabetic

The dataset contains:

- Class 0: **500 samples (65.10%)**
- Class 1: **268 samples (34.90%)**

---

# Exploratory Data Analysis

Several exploratory data analysis techniques were performed to understand the dataset.

## Dataset Structure

The dataset was examined to identify:

- Number of rows and columns
- Feature names
- Data types
- Missing values
- Duplicate records
- Target distribution

No duplicate records were identified in the dataset.

## Missing Value Analysis

The dataset did not contain explicit `NaN` values.

However, several medical features contained zero values that are physiologically implausible in this context.

The following features were examined:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

These zero values were treated as invalid or missing observations.

## Class Distribution

The target variable contains 500 non-diabetic observations and 268 diabetic observations.

This indicates that the dataset is not perfectly balanced.

## Univariate Analysis

Univariate analysis was performed using visualizations and descriptive statistics to understand the distributions of individual features.

The analysis included features such as:

- Glucose
- BMI
- Age
- Blood Pressure
- Pregnancies
- Insulin

## Bivariate Analysis

Relationships between selected features and the target variable were investigated using:

- Box plots
- Violin plots
- Scatter plots
- Grouped statistics

The following relationships were analyzed:

- Glucose vs Outcome
- BMI vs Outcome
- Age vs Outcome
- Blood Pressure vs Outcome
- Pregnancies vs Outcome

## Correlation Analysis

Correlation analysis was performed to examine the association between the input features and the diabetes outcome.

Some observed correlations with the target were:

| Feature | Correlation |
|---|---:|
| Glucose | 0.4666 |
| BMI | 0.2927 |
| Age | 0.2384 |
| Pregnancies | 0.2219 |
| DiabetesPedigreeFunction | 0.1738 |
| Insulin | 0.1305 |
| BloodPressure | 0.0651 |
| SkinThickness | 0.0748 |

Glucose showed the strongest association with the target among the original input features.

Correlation values were used for analysis and feature selection, not as evidence of causation.

## Outlier Analysis

Box plots and descriptive statistics were used to identify potentially extreme observations.

Extreme values were examined rather than automatically removed because some medical measurements may represent genuine observations.

---

# Data Preprocessing

## Invalid Value Handling

Zero values in the following features were treated as invalid or missing:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

These values were replaced with missing values before imputation.

## Median Imputation

Median values were used to replace the invalid values.

The median values used during the project were:

| Feature | Median |
|---|---:|
| Glucose | 117.0 |
| BloodPressure | 72.0 |
| SkinThickness | 29.0 |
| Insulin | 125.0 |
| BMI | 32.3 |

The same preprocessing logic was also incorporated into the deployed application.

---

# Feature Engineering

Additional features were created to capture useful patterns in the data.

## Age Groups

Age was grouped into categories:

- Young
- Adult
- Middle_Age
- Senior

## BMI Categories

BMI was categorized as:

- Underweight
- Normal
- Overweight
- Obese

## Glucose Categories

Glucose values were grouped into:

- Normal
- Prediabetic
- High

## Interaction Features

The following interaction features were created:

- `Glucose_BMI`
- `Glucose_Age`
- `BMI_Age`

## Log-Transformed Features

Log transformations were applied to:

- `Insulin`
- `DiabetesPedigreeFunction`

The resulting features were:

- `Insulin_Log`
- `DiabetesPedigreeFunction_Log`

---

# Final Modeling Features

The final numerical features used for model development were:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Glucose_BMI
Glucose_Age
BMI_Age
Insulin_Log
DiabetesPedigreeFunction_Log
