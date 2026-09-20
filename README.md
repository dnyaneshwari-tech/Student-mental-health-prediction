Student Mental Health Prediction

A machine learning regression project that predicts a student's Mental
Health Score using academic, social-media usage, lifestyle, and
stress-related factors.

Project Overview

This project analyzes a dataset containing information about students'
social media usage, academic habits, physical activity, sleep, stress
level, and mental health score.

The workflow includes:

Data loading and exploration

Data quality checking

Duplicate detection and removal

Outlier analysis

Data cleaning

Exploratory Data Analysis (EDA)

Feature preprocessing

Linear Regression

Random Forest Regression

Randomized hyperparameter tuning

Model evaluation

Model serialization using Joblib

The final model is used in a web application through a FastAPI backend.

Important: This project predicts a numerical mental health score
from the dataset. It is a machine learning demonstration and should
not be treated as a medical or psychological diagnosis.

Dataset

Dataset: Student Social Media And Mental Health Impact.csv

The original dataset contains:

5,000 rows

13 columns

Features

Feature                     Description

Age                       Student age
Gender                    Student gender
Country                   Student's country
Academic_Level            Academic level
Most_Used_Platform        Most-used social media platform
Purpose_Of_Use            Main purpose of social media use
Avg_Daily_Usage_Hours     Average daily social media usage
Daily_Unlocks             Number of daily device/app unlocks
Study_Hours               Daily study hours
Physical_Activity_Hours   Physical activity hours
Sleep_Hours_Per_Night     Hours of sleep per night
Stress_Level              Reported stress level
Mental_Health_Score       Target variable

Target Variable

The target variable is:

Mental_Health_Score

This is a continuous numerical value, so the problem is treated as a
regression problem rather than classification.

Data Cleaning

The notebook performs several data-quality checks.

Missing Values

The initial dataset contains no missing values.

Duplicate Records

The dataset contains 2 duplicate rows.

These duplicates are removed using:

df = df.drop_duplicates()

This leaves 4,998 records for modeling.

Physical Activity Correction

The notebook identifies negative values in Physical_Activity_Hours and
clips them to zero:

df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0)

Outlier Analysis

The Interquartile Range (IQR) method is used to inspect numerical
outliers.

The notebook does not remove all detected outliers. Instead, the
negative physical-activity values are corrected and duplicate rows are
removed.

Exploratory Data Analysis

The notebook explores the dataset using:

Histograms

Correlation heatmap

Social media platform distribution

Stress-level boxplots

Social media usage vs. mental health score

Sleep duration vs. mental health score

Numerical feature distributions

Skewness analysis

Some of the relationships investigated include:

Average daily social media usage vs. mental health score

Sleep hours vs. mental health score

Stress level vs. mental health score

Distribution of commonly used social media platforms

Feature Engineering and Preprocessing

The features are divided into different groups.

Skewed Numerical Feature

Study_Hours is processed using:

Log1p transformation → StandardScaler

Other Numerical Features

The following features are standardized:

Age

Avg_Daily_Usage_Hours

Daily_Unlocks

Physical_Activity_Hours

Sleep_Hours_Per_Night

Ordinal Feature

Stress_Level has a natural order:

Low → Medium → High → Very High

It is encoded using OrdinalEncoder.

Categorical Features

The following categorical features are encoded using OneHotEncoder:

Gender

Academic_Level

Most_Used_Platform

Purpose_Of_Use

Unknown categories are handled using:

OneHotEncoder(handle_unknown="ignore")

Country Grouping

The notebook identifies the 10 most frequent countries and groups all
remaining countries into Other.

The resulting categories are:

Other

India

USA

Canada

Australia

UK

Germany

Mexico

Turkey

France

Train-Test Split

The dataset is divided into:

70% training data

30% testing data

using:

train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

Machine Learning Models

1. Linear Regression

A preprocessing pipeline is combined with Linear Regression.

Testing results:

R²: 0.7344

MAE: 0.5425

RMSE: 0.6830

2. Random Forest Regression

A Random Forest Regressor is combined with the same preprocessing
pipeline.

Testing results:

R²: 0.8672

MAE: 0.3668

RMSE: 0.4830

Training R²:

0.9796

3. Tuned Random Forest

RandomizedSearchCV is used for hyperparameter tuning.

The search uses:

15 random parameter combinations

5-fold cross-validation

R² scoring

Parallel processing with n_jobs=-1

Best parameters found:

n_estimators = 300
max_depth = 15
min_samples_split = 5
min_samples_leaf = 2

Testing results:

R²: 0.8564

MAE: 0.3843

RMSE: 0.5023

Model Comparison

Model                       Test R²   Training R²      MAE     RMSE

Linear Regression            0.7344        0.7135   0.5425   0.6830
Random Forest (default)      0.8672        0.9796   0.3668   0.4830
Random Forest (tuned)        0.8564        0.9566   0.3843   0.5023

The notebook serializes the default Random Forest pipeline as the
final model used by the application:

joblib.dump(rf_pipeline, 'Mental_Health_Model.pkl')

Saved Model

The trained model is saved as:

Mental_Health_Model.pkl

The saved object contains the preprocessing pipeline and Random Forest
model, allowing the application to receive raw input features and
perform the required preprocessing before prediction.

Project Structure

Student-mental-health-prediction/
│
├── index.html
├── style.css
├── script.js
├── main.py
├── Mental_Health_Model.pkl
├── Student Social Media And Mental Health Impact.csv
├── Student mental health prediction.ipynb
└── requirements.txt

Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Joblib

FastAPI

HTML

CSS

JavaScript

Machine Learning Pipeline

Raw Student Data
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Preprocessing
       ↓
Train/Test Split
       ↓
Random Forest Regression
       ↓
Model Evaluation
       ↓
Joblib Model Serialization
       ↓
FastAPI Backend
       ↓
Web Application

Web Application

The trained model is integrated into a web application.

The application allows users to provide student-related information and
receive a predicted Mental Health Score through the trained machine
learning model.

The frontend communicates with the FastAPI backend, which loads:

Mental_Health_Model.pkl

and performs the prediction.

How to Run the Project Locally

1. Clone the repository

git clone https://github.com/dnyaneshwari-tech/Student-mental-health-prediction.git
cd Student-mental-health-prediction

2. Install dependencies

pip install -r requirements.txt

3. Start the FastAPI backend

uvicorn main:app --reload

4. Open the frontend

Open:

index.html

in a browser, or use a local development server such as VS Code Live
Server.

Evaluation Metrics

R² Score

R² measures how much of the variation in the target variable is
explained by the model.

Higher values indicate that the model explains more of the variation in
the target data.

Mean Absolute Error (MAE)

MAE measures the average absolute difference between the actual and
predicted mental health scores.

Lower values indicate smaller prediction errors.

Root Mean Squared Error (RMSE)

RMSE measures prediction error while giving larger errors more weight.

Lower values indicate smaller prediction errors.

Limitations

The model is trained on the provided dataset and its patterns may
not generalize to other populations.

The dataset contains self-reported lifestyle and stress-related
information.

A predicted score should not be interpreted as a clinical diagnosis.

Model performance depends on the quality and distribution of the
training data.

The tuned Random Forest produced a lower test R² than the default
Random Forest in this experiment, so the notebook saves the default
Random Forest pipeline as the deployed model.

Future Improvements

Possible improvements include:

Testing additional regression algorithms

More systematic feature engineering

Cross-validation comparison of multiple models

Feature importance and explainability

Improved validation on external data

Monitoring model performance after deployment

Adding model versioning

Improving API response handling and deployment performance

Author

Dnyaneshwari

Artificial Intelligence & Data Science Student

⭐ If you find this project useful, consider giving the repository a
star.
