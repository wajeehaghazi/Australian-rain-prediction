
Australian Rain Prediction Using Machine Learning
Overview

This project predicts whether it will rain tomorrow in Australia using historical weather observations and machine learning techniques. The objective is to build an end-to-end machine learning pipeline that performs data preprocessing, feature engineering, model training, hyperparameter tuning, model selection, and deployment through a FastAPI prediction service.

The project uses the Australian Weather Dataset and compares multiple classification algorithms to identify the best-performing model for rainfall prediction.

Problem Statement

Weather forecasting plays a critical role in agriculture, transportation, disaster management, and daily decision-making. Predicting rainfall accurately can help individuals and organizations prepare for weather-related events.

This project aims to answer the following question:

"Will it rain tomorrow based on today's weather conditions?"

Dataset

Dataset: Australian Weather Dataset

Source:
https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

The dataset contains historical weather observations collected from multiple locations across Australia, including:

Temperature
Rainfall
Humidity
Wind Direction
Wind Speed
Atmospheric Pressure
Cloud Cover
Sunshine Hours
Rain Indicators

Target Variable:

RainTomorrow
Yes → Rain expected tomorrow
No → No rain expected tomorrow
Project Workflow
Data Collection
       ↓
Data Cleaning
       ↓
Missing Value Handling
       ↓
Feature Engineering
       ↓
Feature Scaling
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Hyperparameter Tuning
       ↓
Model Comparison
       ↓
Best Model Selection
       ↓
Model Persistence
       ↓
FastAPI Deployment
Data Preprocessing

The following preprocessing techniques were applied:

Missing Value Handling
Numerical features were filled using mean imputation.
Categorical features were filled using mode imputation.
Feature Encoding
RainTomorrow encoded using LabelEncoder
RainToday mapped to binary values
Wind direction features encoded using OrdinalEncoder
Location encoded using LabelEncoder
Feature Scaling

StandardScaler was applied to numerical features to normalize feature distributions before model training.

Models Implemented
Logistic Regression

A linear classification algorithm used as a baseline model.

Hyperparameter tuning performed using:

GridSearchCV

Parameters tuned:

C
Solver
Penalty
Decision Tree Classifier

A tree-based classification algorithm capable of learning nonlinear decision boundaries.

Hyperparameter tuning performed using:

GridSearchCV

Parameters tuned:

Max Depth
Min Samples Split
Min Samples Leaf
Criterion
Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Hyperparameter tuning performed using:

RandomizedSearchCV

Parameters tuned:

Number of Estimators
Max Depth
Min Samples Split
Min Samples Leaf
Max Features
Model Performance
Model	Accuracy
Logistic Regression	83.65%
Decision Tree	83.01%
Random Forest	83.91%
Best Model
Random Forest Classifier

Best Accuracy:

83.91%
Model Evaluation

Evaluation metrics used:

Accuracy Score
Confusion Matrix
Classification Report
ROC Curve
Feature Importance Analysis
Model Persistence

The best-performing model is automatically saved after training.

Saved artifacts:

models/
├── best_rain_model.pkl
├── scaler.pkl

This allows the model to be reused without retraining.
