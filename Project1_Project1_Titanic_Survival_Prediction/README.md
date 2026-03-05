# 🚢 Project 1: Titanic Survival Prediction — Binary Classification

## Overview
Predicted passenger survival on the Titanic using machine learning classification models. This is a foundational supervised learning project covering the full ML pipeline — from raw data exploration to model evaluation and comparison.

## Business / Learning Context
The Titanic dataset is a classic binary classification problem. The goal is to predict whether a passenger survived (1) or did not survive (0) based on features like age, gender, ticket class, and family size. This project demonstrates core skills in EDA, feature engineering, handling missing data, and model selection.

## Dataset
- **Source:** [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- **Size:** 891 training rows × 12 columns
- **Target:** `Survived` (0 = No, 1 = Yes)

| Column | Description |
|---|---|
| Pclass | Ticket class (1st, 2nd, 3rd) |
| Sex | Gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Passenger fare |
| Embarked | Port of embarkation |

## Approach

**Step 1 — Exploratory Data Analysis (EDA)**
- Survival rate analysis by gender, class, age group
- Correlation heatmap and distribution plots
- Identified missing values: Age (19.8%), Cabin (77%), Embarked (0.2%)

**Step 2 — Data Preprocessing & Feature Engineering**
- Imputed Age using median by Pclass and Sex groups
- Dropped Cabin column (too many missing values)
- Created new features: `FamilySize` (SibSp + Parch + 1)
- Encoded categorical variables (Sex, Embarked) using Label Encoding

**Step 3 — Model Training & Comparison**
- Logistic Regression (baseline)
- Random Forest Classifier
- Cross-validation (5-fold) for both models

**Step 4 — Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualisation
- ROC Curve and AUC score

## Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 81% | 0.88 |
| **Random Forest (Best)** | **83%** | **0.89** |

Key insight: Gender was the strongest predictor of survival, followed by Fare, Pclass and engineered Title feature.

## Key Skills Demonstrated
- Binary classification
- Exploratory Data Analysis (EDA)
- Missing value imputation strategies
- Feature engineering from raw text
- Model comparison and evaluation metrics
- Confusion matrix interpretation

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn`

## How to Run
1. Download `train.csv` from [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data)
2. Open `Project1_Titanic_Survival_Prediction.ipynb` in [Google Colab](https://colab.research.google.com)
3. Upload `train.csv` via the Files panel
4. Run all cells top to bottom (`Runtime → Run all`)

## Files in This Folder
```
Project1_Titanic_Survival_Prediction/
├── Project1_Titanic_Survival_Prediction.ipynb   ← Main notebook
└── README.md                                     ← This file
```
> Dataset included or download free from Kaggle link above
