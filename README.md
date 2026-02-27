# Loan-risk-prediction-analysis
End-to-end Machine Learning project to predict loan eligibility using historical customer application data. This project implements comprehensive data cleaning, feature engineering through One-Hot Encoding, and predictive modeling to automate risk assessment in financial lending.

### Project Overview
The objective of this project is to automate the loan eligibility process in real-time based on customer details such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, and Credit History.

### Database Description
The dataset contains 614 observations and 13 features related to customer loan applications. It includes a mix of categorical and numerical information required for risk assessment.
|Feature Name|Description|Data Type|
|------------|-----------|---------|
|Loan_ID|Unique identifier for each application|Unique identifier for each application|
|Gender|Applicant's gender (Male/Female)|Categorical|
|Married|Marital status (Married/Single)|Categorical|
|Dependents|Number of dependents (0, 1, 2, 3+)|Categorical/Ordinal|
|Education|Applicant's education (Graduate/Not Graduate)|Categorical|
|Self_Employed|Employment status (Yes/No)|Categorical|
|ApplicantIncome|Income of the primary applicant|Numerical (int64)|
|CoapplicantIncome|Income of the co-applicant|Numerical (float64)|
|LoanAmount|Loan amount in thousands|Numerical (float)|
|Loan_Amount_Term|Term of the loan in months|Numerical (float)|
|Credit_History|Credit history meets guidelines (1.0 = Yes, 0.0 = No)|Numerical|
|Property_Area|Area of property (Urban/Semiurban/Rural)|Categorical|
|Loan_Status|Loan approved (Y/N)|Categorical (Binary)|




### Technical Stack
1. Language: Python
2. Environment: Jupyter notebook
3. Libraries: Pandas (Data Manipulation), Scikit-Learn (Modeling & Evaluation), Matplotlib (Modeling)

### Workflow Steps
Data Cleaning: Identified and handled missing values across multiple features (Gender, Dependents, LoanAmount, etc.) using forward and backward filling techniques to ensure data integrity.
Feature Engineering: Performed One-Hot Encoding on categorical variables including Gender, Married, Education, and Property_Area.
Transformed the target variable Loan_Status into a machine-readable format.
Modeling: Split the dataset into training and testing sets to build a robust classification model.
Performance Metrics: Evaluated the model using a Confusion Matrix and achieved an Accuracy Score of ~85%.

### Key insights
1. Credit History is the strongest predictor: Applicants with a Credit_History of 1.0 (meeting guidelines) had a significantly higher probability of loan approval compared to those with 0.0.
2. Income Distribution: The average ApplicantIncome is approximately $5,403, but the data is highly skewed with a maximum value of $81,000, suggesting the presence of high-income outliers.
3. Education Impact: The majority of applicants (~78%) are Graduates, which correlates with higher loan approval rates in this specific demographic.
4. Property Trends: Applicants from Semiurban areas were the most frequent in the dataset (233 cases), indicating a high demand for financing in developing residential zones.
5. Approval Rate: The baseline dataset shows that approximately 68% of loans (422 out of 614) were approved, which provided a balanced foundation for the classification model.
6. Model Accuracy: The final classification pipeline achieved an accuracy of ~85.06%, indicating high reliability for automated risk filtering.


