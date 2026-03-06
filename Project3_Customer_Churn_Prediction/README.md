# 📉 Project 3: Customer Churn Prediction — Binary Classification

## Overview
Built a machine learning pipeline to predict which telecom customers are likely to churn (cancel their subscription). This project tackles one of the most commercially valuable classification problems in business — with direct applications in telecoms, banking, SaaS, and any subscription-based industry.

## Business / Learning Context
Customer churn costs companies significantly more than retaining existing customers. By identifying at-risk customers before they leave, businesses can take proactive retention actions — targeted offers, service improvements, or account manager outreach. This project demonstrates class imbalance handling, business-focused threshold tuning, and AUC-ROC evaluation — skills directly applicable to any risk scoring or early-warning system.

## Dataset
- **Source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 rows × 21 columns
- **Target:** `Churn` (Yes / No → encoded as 1 / 0)
- **Class imbalance:** ~73% No Churn, ~27% Churn

Key features:
| Column | Description |
|---|---|
| tenure | Months the customer has been with the company |
| MonthlyCharges | Current monthly charge amount |
| TotalCharges | Total amount charged to date |
| Contract | Month-to-month, One year, Two year |
| InternetService | DSL, Fiber optic, No |
| TechSupport | Yes, No, No internet service |
| PaymentMethod | Electronic check, Mail check, etc. |

## Approach

**Step 1 — EDA**
- Churn rate analysis by contract type, tenure group, monthly charges
- Key finding: Month-to-month contract customers churn at 3× the rate of annual contract customers


**Step 2 — Preprocessing**
- Converted `TotalCharges` from object to numeric (blank strings → NaN → imputed)
- Label encoding for binary categoricals (Yes/No columns)
- One-hot encoding for multi-class categoricals (Contract, InternetService, PaymentMethod)
- Standard scaling for numeric features (tenure, MonthlyCharges, TotalCharges)

**Step 3 — Model Training & Comparison**
- Logistic Regression
- Random Forest Classifier


**Step 4 — Evaluation**
- Confusion Matrix, Classification Report
- AUC-ROC curve and score
- Precision, Recall, F1-Score at default and optimised thresholds

**Step 5 — Business Threshold Analysis**
- Default threshold is 0.5 — but for churn prediction, it may be better to lower the threshold to catch more at-risk customers (higher recall), even at the cost of some precision

## Results

| Model | AUC-ROC | Recall (Churn) | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.84 | 0.57 | 0.61 |
| Random Forest | 0.83 | 0.50 | 0.56 |
| Logistic Regression (Threshold changed) | 0.84 | 0.71 | 0.62 |
| Random Forest (Tuned) | 0.84 | 0.49 | 0.57 |
| **Random Forest (Threshold changed)** | **0.84** | **0.72** | **0.63** |

Key insight: tenure and TotalCharges are the strongest churn predictors. Customers on month-to-month contracts with less than 12 months tenure are the highest-risk segment.

## Key Skills Demonstrated
- Binary classification on real-world imbalanced data
- Business-oriented threshold tuning (Precision vs Recall tradeoff)
- AUC-ROC evaluation
- Feature importance analysis
- Actionable business insights from model outputs

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn`  `Matplotlib` `Seaborn`

## How to Run
1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Open `Project3_Customer_Churn_Prediction.ipynb` in [Google Colab](https://colab.research.google.com)
3. Upload the CSV via the Files panel
4. Run all cells top to bottom (`Runtime → Run all`)

## Files in This Folder
```
Project3_Customer_Churn_Prediction/
├── Project3_Customer_Churn_Prediction.ipynb   ← Main notebook
└── README.md                                   ← This file
```
> Dataset included or download free from Kaggle link above
