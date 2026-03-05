# 🏠 Project 2: House Price Prediction — Regression

## Overview
Built a regression model to predict residential property sale prices using the Ames Housing dataset — one of the most feature-rich real estate datasets available, with 80+ variables describing every aspect of residential homes. This project covers advanced feature engineering, handling skewed distributions, and gradient boosting regression.

## Business / Learning Context
Accurate property price prediction is used by real estate platforms, banks (mortgage risk), and investment firms. This project demonstrates regression modelling skills applicable to any domain where a continuous numerical outcome must be predicted from structured data — including financial forecasting, cost estimation, and revenue prediction in a business analytics context.

## Dataset
- **Source:** [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Size:** 1,460 training rows × 81 columns
- **Target:** `SalePrice` (sale price of the house in USD)

Key feature categories:
- Location and zoning (Neighborhood, MSZoning)
- Size and area (GrLivArea, TotalBsmtSF, GarageArea)
- Quality ratings (OverallQual, OverallCond, KitchenQual)
- Age and condition (YearBuilt, YearRemodAdd)
- Extras (PoolArea, Fireplaces, GarageType)

## Approach

**Step 1 — EDA**
- Distribution analysis of SalePrice (right-skewed)
- Correlation analysis — OverallQual, GrLivArea, GarageCars strongest correlates
- Scatter plots and boxplots for key numeric and categorical features

**Step 2 — Data Preprocessing**
- Handled 19 columns with missing values — different strategy per column (median, mode, "None" for categorical)
- Log transformation of SalePrice and skewed numeric features (reduces impact of outliers)


**Step 3 — Feature Engineering**
- One-hot encoding for all categorical variables

**Step 4 — Model Training**
- Linear Regression (baseline)
- Ridge Regression (L2 regularisation)
- Lasso Regression
- Random Forest (best model)
- 5-fold cross-validation throughout

**Step 5 — Evaluation**
- RMSE on log-transformed predictions
- R² score
- Residual plots to check model assumptions

## Results

| Model | RMSE (log scale) | R² |
|---|---|---|
| Linear Regression (baseline) | 0.208 | 0.77 |
| Ridge Regression | 0.198 | 0.79 |
| Lasso Regression | 0.170 | 0.84 |
| **Random Forest (Best)** | **0.14** | **0.88** |

**33% RMSE improvement** achieved by Random Forest over linear baseline.  
Top predictors: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt.

## Key Skills Demonstrated
- Regression modelling with Random Forest
- Handling missing data at scale (19 columns)
- Log transformation and feature scaling
- Feature engineering from domain knowledge
- Regularisation (Ridge)
- Cross-validation and RMSE evaluation

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn`

## How to Run
1. Download dataset from [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. Open `Project2_House_Price_Prediction.ipynb` in [Google Colab](https://colab.research.google.com)
3. Upload `train.csv` via the Files panel
4. Run all cells top to bottom (`Runtime → Run all`)

## Files in This Folder
```
Project2_House_Price_Prediction/
├── Project2_House_Price_Prediction.ipynb   ← Main notebook
└── README.md                                ← This file
```
> Dataset included or download free from Kaggle link above
