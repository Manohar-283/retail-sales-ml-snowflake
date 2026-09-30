# End-to-End Retail Sales Prediction ML Workflow Using Snowflake

## Project Overview

This project implements an end-to-end machine learning workflow for **daily retail sales prediction using Snowflake and XGBoost**.

The project uses the **UCI Online Retail dataset** and moves through the complete ML lifecycle:

**Data ingestion → Data cleaning → Daily aggregation → Feature engineering → Time-series split → XGBoost training → Model evaluation → Model Registry → Warehouse inference → Prediction storage**

The goal is to demonstrate how Snowflake can be used as the central platform for both data preparation and machine learning.

---

## Problem Statement

Retail businesses need to understand and predict daily sales to support inventory planning, sales forecasting, and operational decision-making.

The original transaction-level sales value can be calculated directly from quantity and unit price. Therefore, this project instead formulates a more meaningful ML problem:

> **Predict the total sales amount for a day using historical sales patterns and calendar features.**

---

## Dataset

**Dataset:** UCI Online Retail Dataset

The dataset contains transaction-level retail information including:

- Invoice number
- Stock code
- Product description
- Quantity
- Invoice date
- Unit price
- Customer ID
- Country

The original CSV was uploaded into Snowflake as:

`ONLINE_RETAIL`

The dataset contains **541,909 transaction records**.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Snowflake | Data storage, SQL transformation and ML workflow |
| Snowflake Notebooks | Python-based ML development |
| Snowpark Python | Accessing Snowflake data from Python |
| XGBoost | Regression model |
| Pandas | Data manipulation |
| Scikit-learn | Model evaluation |
| GitHub | Version control and project hosting |

---

## Project Architecture

```text
                 UCI Online Retail Dataset
                           |
                           v
                  +------------------+
                  |     Snowflake    |
                  |  ONLINE_RETAIL   |
                  +------------------+
                           |
                           v
                  Data Cleaning
                           |
                           v
                  RETAIL_SALES_CLEAN
                           |
                           v
                Daily Sales Aggregation
                           |
                           v
                 DAILY_RETAIL_SALES
                           |
                           v
                 Feature Engineering
                           |
                           v
                 RETAIL_FORECAST_DATA
                           |
                           v
                    RETAIL_ML_FINAL
                           |
                           v
                    Time-Series Split
                       /        \
                      /          \
                 Training        Testing
                    |               |
                    v               v
                 XGBoost       Model Evaluation
                    |
                    v
             Snowflake Model Registry
                    |
                    v
             Warehouse Inference
                    |
                    v
          RETAIL_SALES_PREDICTIONS
```

---

## Snowflake Setup

### Database

`RETAIL_ML_DB`

### Schema

`RETAIL_SCHEMA`

### Warehouse

`RETAIL_ML_WH`

The warehouse was configured as an X-Small warehouse with automatic suspension and resume.

---

## Data Preparation

### 1. Transaction Sales Amount

A transaction-level sales amount was calculated as:

```text
SALES_AMOUNT = QUANTITY × UNITPRICE
```

This was stored in:

`RETAIL_SALES`

### 2. Data Cleaning

Transactions with:

- `QUANTITY <= 0`
- `UNITPRICE <= 0`

were removed.

The cleaned data is stored in:

`RETAIL_SALES_CLEAN`

This resulted in **530,100 valid transactions**.

### 3. Daily Aggregation

Transactions were aggregated by date to produce:

- `TOTAL_SALES`
- `TOTAL_QUANTITY`
- `TRANSACTION_COUNT`

Stored in:

`DAILY_RETAIL_SALES`

The available daily data covers:

**2010-12-01 to 2011-12-09**

with **305 daily records**.

---

## Feature Engineering

The ML dataset uses the following features:

| Feature | Description |
|---|---|
| `YEAR` | Year of the sales date |
| `MONTH` | Month |
| `DAY` | Day of month |
| `DAY_OF_WEEK` | Day of week |
| `SALES_LAG_1` | Previous day's sales |
| `SALES_LAG_7` | Sales from seven days earlier |
| `SALES_LAG_30` | Sales from thirty days earlier |
| `SALES_ROLLING_7` | Previous 7-day average sales |

Target variable:

```text
TOTAL_SALES
```

The final ML dataset is:

`RETAIL_ML_FINAL`

After removing rows without sufficient historical lag information, the final dataset contains **275 rows**.

---

## Train/Test Split

Because this is a time-series prediction problem, the data is split chronologically rather than randomly.

```text
Total records : 275
Training      : 220
Testing       : 55
```

The first 80% of the time-ordered records are used for training and the remaining 20% for testing.

---

## Machine Learning Model

### XGBoost Regressor

The model uses:

```python
XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
```

The model predicts:

```text
TOTAL_SALES
```

from the engineered time-series features.

---

## Model Evaluation

The model is evaluated using:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted sales.

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly.

### R² — R-squared

Measures how much of the variation in the target is explained by the model.

The exact metric values are generated in the Snowflake notebook during execution.

---

## Snowflake Model Registry

The trained XGBoost model is registered in Snowflake Model Registry as:

```text
RETAIL_SALES_XGBOOST
```

Warehouse inference is enabled using version:

```text
V2
```

This allows the registered model to be used for inference from Snowflake.

---

## Prediction Output

Predictions are stored in:

```text
RETAIL_SALES_PREDICTIONS
```

The output contains:

- `TOTAL_SALES` — actual sales
- `PREDICTED_SALES` — model prediction

This table can be used for further visualization and dashboard development in Snowflake.

---

## Project Workflow

```text
1. Upload UCI Online Retail CSV
2. Create Snowflake database and schema
3. Create Snowflake warehouse
4. Clean transaction data
5. Calculate sales amount
6. Aggregate transactions by day
7. Create lag and rolling features
8. Create final ML dataset
9. Split data chronologically
10. Train XGBoost model
11. Evaluate MAE, RMSE and R²
12. Register model in Snowflake Model Registry
13. Run warehouse inference
14. Store predictions in Snowflake
```

---

## Repository Structure

```text
retail-sales-ml-snowflake/
│
├── 01_setup.sql
├── RETAIL_SALES_ML.py
├── README.md
└── requirements.txt
```

---

## How to Run

### 1. Snowflake

Create the database, schema and warehouse and execute the SQL transformations in:

```text
01_setup.sql
```

Upload the UCI Online Retail CSV as `ONLINE_RETAIL`.

### 2. Python Environment

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the ML Workflow

Open:

```text
RETAIL_SALES_ML.py
```

The workflow:

- Connects to the active Snowflake session
- Loads `RETAIL_ML_FINAL`
- Creates training and testing datasets
- Trains XGBoost
- Calculates evaluation metrics
- Registers the model
- Performs inference
- Saves predictions to Snowflake

---

## Important Note

The project intentionally uses **daily sales forecasting** instead of directly predicting transaction-level sales.

This avoids a trivial target relationship where:

```text
SALES_AMOUNT = QUANTITY × UNITPRICE
```

and makes the machine learning problem more representative of a forecasting workflow.

---

## Future Enhancements

- Build an interactive Streamlit dashboard
- Add product/category-level forecasting
- Add customer segmentation
- Compare XGBoost with Random Forest and other forecasting approaches
- Add automated model retraining
- Add Snowflake tasks and streams for scheduled pipelines
- Add model monitoring
- Deploy predictions to a business dashboard

---

## Author

**Raja Manohar**

B.Tech — Computer Science and Engineering (AI & ML)

Anil Neerukonda Institute of Technology and Sciences (ANITS)

---

## License

This project is intended for educational and portfolio purposes.
