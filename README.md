# LoanTap Credit Risk Assessment Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)  
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)  
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Graphics-teal)  
![Statsmodels](https://img.shields.io/badge/Statsmodels-Statistical%20Modeling-purple)  
![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-Classification-brown)  
![Credit Risk](https://img.shields.io/badge/Credit%20Risk-Default%20Prediction-crimson)

---

## Overview
This project implements a **logistic regression model** to assess credit risk for LoanTap, a digital lending platform. The goal is to predict whether a loan applicant will **default** (fail to repay) based on their application data, enabling better lending decisions and reduced financial risk.

## Problem Statement
LoanTap requires an automated system to:
- Evaluate loan applications efficiently  
- Identify high-risk applicants likely to default  
- Minimize defaults while maintaining customer acquisition  

## Solution Approach
We developed a binary classification model using **logistic regression** to predict loan default probability. The model leverages applicant demographic, financial, and behavioral data to generate risk scores for informed lending decisions.

## Project Structure
```
loan-risk-assessment/
├── Project 9 - Business Case - LoanTap - Logistic Regression.ipynb  # Main analysis notebook
├── data/                          # Raw and processed datasets
│   ├── loan_data.csv              # Original dataset
│   └── cleaned_data.csv           # Processed dataset
├── models/                        # Trained model artifacts
│   └── logistic_regression_model.pkl
├── README.md                      # This documentation file
└── requirements.txt               # Project dependencies
```

## Key Features
- **Data Preprocessing**: Handling missing values, outlier treatment, and feature engineering  
- **Exploratory Data Analysis (EDA)**: Comprehensive visualizations of default patterns  
- **Feature Selection**: Statistical methods to identify most predictive variables  
- **Model Training**: Logistic regression with hyperparameter tuning  
- **Model Evaluation**: Metrics including accuracy, precision, recall, F1-score, and ROC-AUC  
- **Business Interpretation**: Translating model outputs into actionable lending policies  

## Technical Requirements
- Python 3.8+  
- Jupyter Notebook  
- Key libraries:  
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib  
  - seaborn  
  - statsmodels  

## Installation
1. Clone this repository  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```

## Usage
1. Open `Project 9 - Business Case - LoanTap - Logistic Regression.ipynb`  
2. Execute cells sequentially to:  
   - Load and explore the dataset  
   - Preprocess the data  
   - Train the logistic regression model  
   - Evaluate model performance  
   - Generate predictions for new applicants  

## Model Performance
- **Accuracy**: [Insert final accuracy]  
- **Precision**: [Insert precision score]  
- **Recall**: [Insert recall score]  
- **F1-Score**: [Insert F1-score]  
- **ROC-AUC**: [Insert AUC score]  

*Note: Actual metrics available in the notebook results section*

## Business Impact
- Potential to **reduce default rates** by [X]%  
- Enable **automated decisioning** for [Y]% of applications  
- Improve **risk-adjusted returns** on loan portfolio  
- Support **scalable lending operations** with consistent risk assessment  

## Future Improvements
- Test ensemble methods (Random Forest, XGBoost)  
- Incorporate alternative data sources (cash flow, social signals)  
- Implement real-time scoring API  
- Add model monitoring for concept drift detection  

## License
This project is for educational and demonstration purposes only. The dataset and model are not for production use without proper validation and regulatory compliance.

---

*Developed as part of a business analytics case study*  
*Last updated: November 2025*
