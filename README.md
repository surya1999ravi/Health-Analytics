# Health-Analytics
Heart Disease Risk Stratification – Healthcare Analytics Project
Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction and risk stratification can help healthcare professionals identify high-risk patients and provide preventive treatment.

This project focuses on analyzing healthcare data and building a machine learning model to predict the likelihood of heart disease based on patient health parameters. The project uses Python-based data analysis, machine learning techniques, and a Streamlit application to provide predictions in an interactive way.

The objective is to demonstrate how data analytics and machine learning can support healthcare decision-making.

Dataset Source

The dataset used for this project is a Heart Disease dataset available from public machine learning repositories.

Dataset Source:

Kaggle – Heart Disease Dataset

UCI Machine Learning Repository – Heart Disease Dataset

The dataset contains medical attributes of patients that help determine whether a person is at risk of heart disease.

Key Features in the Dataset
Feature	Description
Age	Age of the patient
Sex	Gender of the patient
ChestPainType	Type of chest pain experienced
RestingBP	Resting blood pressure
Cholesterol	Serum cholesterol level
FastingBS	Fasting blood sugar level
RestingECG	Resting electrocardiogram results
MaxHR	Maximum heart rate achieved
ExerciseAngina	Exercise induced angina
Oldpeak	ST depression induced by exercise
ST_Slope	Slope of peak exercise ST segment
HeartDisease	Target variable indicating presence of heart disease
Project Workflow

The project follows a structured data science pipeline:

Data Collection

Data Cleaning and Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Model Building

Model Evaluation

Deployment using Streamlit

Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset and identify patterns related to heart disease risk.

Key analyses performed include:

Distribution of heart disease cases

Age vs heart disease relationship

Correlation analysis between features

These analyses help identify important predictors influencing heart disease risk.

Data Preprocessing

Data preprocessing steps include:

Handling missing values

Encoding categorical variables

Feature scaling using StandardScaler

Train-test data splitting

These steps ensure that the dataset is prepared properly before training machine learning models.

Machine Learning Models Used

The following machine learning models were implemented:

Logistic Regression

Logistic Regression is a statistical classification algorithm used to predict the probability of a binary outcome.

Random Forest Classifier

Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.

Model Evaluation

The models were evaluated using standard classification metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

These metrics help measure the performance of the predictive models.

Technologies Used
Programming Language

Python

Libraries

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Streamlit

Development Tools

Jupyter Notebook

VS Code

Streamlit

Streamlit Application

A Streamlit web application was developed to allow users to input patient health parameters and obtain predictions about heart disease risk.

The application performs the following tasks:

Accepts user health inputs

Applies preprocessing

Uses the trained model to predict heart disease risk

Displays prediction results interactively

Project Outcome

This project demonstrates how machine learning can be applied in healthcare analytics to support early detection of heart disease risk.

The developed system can assist healthcare professionals by providing data-driven insights and risk predictions based on patient medical data.

Future Improvements

Future enhancements may include:

Using deep learning models

Integrating real-time healthcare datasets

Improving model accuracy through feature engineering

Deploying the application on cloud platforms
