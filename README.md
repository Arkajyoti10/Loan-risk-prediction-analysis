# Loan-risk-prediction-analysis
End-to-end Machine Learning project to predict loan eligibility using historical customer application data. This project implements comprehensive data cleaning, feature engineering through One-Hot Encoding, and predictive modeling to automate risk assessment in financial lending.

## Project Overview
The objective of this project is to automate the loan eligibility process in real-time based on customer details such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, and Credit History.

## Database Description
The dataset contains 614 observations and 13 features related to customer loan applications. It includes a mix of categorical and numerical information required for risk assessment.
|Feature Name|Description|Data Type|
|Loan_ID|Unique identifier for each application|Unique identifier for each application|
|Gender|Applicant's gender (Male/Female)|Categorical|



## Technical Stack
1. Language: Python
2. Libraries: Pandas (Data Manipulation), Scikit-Learn (Modeling & Evaluation), Matplotlib (Modeling)

## Key Workflow Steps
Data Cleaning: Identified and handled missing values across multiple features (Gender, Dependents, LoanAmount, etc.) using forward and backward filling techniques to ensure data integrity.
Feature Engineering: Performed One-Hot Encoding on categorical variables including Gender, Married, Education, and Property_Area.
Transformed the target variable Loan_Status into a machine-readable format.
Modeling: Split the dataset into training and testing sets to build a robust classification model.
Performance Metrics: Evaluated the model using a Confusion Matrix and achieved an Accuracy Score of ~85%.


