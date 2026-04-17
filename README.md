# Telecom Customer Lifetime Value (CLTV) Prediction

## Overview

This project predicts Customer Lifetime Value for a telecom company using the **Telco Customer Churn** dataset. CLTV estimation helps telecom providers identify high-value customers, optimize retention strategies, and allocate marketing resources effectively.

## Dataset

The dataset used is the **IBM Watson Analytics Telco Customer Churn** dataset, which contains 7,043 customer records with 21 features including:

- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account info**: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Target**: tenure (used as a proxy for CLTV)

## Methodology

### 1. Data Preprocessing
- Dropped non-predictive `customerID` column
- Converted `TotalCharges` to numeric and handled missing values
- Filtered customers with tenure ≥ 12 months (defined time horizon)
- Applied `StandardScaler` to numerical features and `OneHotEncoder` (drop first) to categorical features via a `ColumnTransformer`

### 2. Exploratory Data Analysis
- Correlation heatmap of numerical features to identify relationships between variables

### 3. Model Training
Five regression models were trained and evaluated using scikit-learn pipelines:

| Model | Configuration |
|---|---|
| Linear Regression | Default |
| Decision Tree Regressor | Default |
| Random Forest Regressor | 100 estimators, random_state=42 |
| Gradient Boosting Regressor | 100 estimators, random_state=42 |
| Neural Network (MLP) | hidden_layer_sizes=(50,), max_iter=1000 |

### 4. Evaluation
Models were evaluated using **5-fold cross-validation** (MSE) and **train-test split** with the following metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Results

### Cross-Validation MSE (lower is better)

| Model | Mean CV MSE |
|---|---|
| Neural Network | -2.44 |
| Random Forest | -3.35 |
| Gradient Boosting | -5.28 |
| Decision Tree | -7.03 |
| Linear Regression | -59.69 |

### Test Set Performance

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression | 51.73 | 7.19 | 5.34 | 0.861 |
| Decision Tree | 5.27 | 2.30 | 1.66 | 0.986 |
| Random Forest | 2.68 | 1.64 | 1.20 | 0.993 |
| Gradient Boosting | 5.09 | 2.26 | 1.71 | 0.986 |
| Neural Network | 2.19 | 1.48 | 1.12 | 0.994 |

The **Neural Network (MLP)** achieved the best performance with an R² of **0.994**, closely followed by the **Random Forest** with an R² of **0.993**. Both significantly outperform Linear Regression.

## Tech Stack

- **Python 3**
- **pandas**, **numpy** — data manipulation
- **scikit-learn** — preprocessing, modeling, evaluation
- **matplotlib**, **seaborn** — visualization

## Usage

1. Clone the repository
2. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project directory (update the file path in the notebook accordingly)
3. Open and run `Telecom Customer Lifetime Value (CLTV) Prediction.ipynb` in Jupyter Notebook or Google Colab

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```
