# Monsoon CreditTech Credit Risk Modelling Assignment

## Candidate Details

**Name:** Lakshya Vipassana

**Program:** M.Tech, Infrastructure Design and Management

**Institute:** Indian Institute of Technology Kharagpur

---

# Problem Statement

The objective is to build a machine learning model that predicts the probability of loan default at the time of application.

The target variable is:

* 1 = Bad Loan (Default)
* 0 = Good Loan

The evaluation metric is:

**ROC-AUC Score**

---

# Dataset Description

The dataset consists of three major components:

### 1. Train/Test Flag Data

Contains:

* uid
* TARGET (available only for train)

### 2. Accounts Data

Historical credit accounts of borrowers:

* credit_type
* loan_amount
* amount_overdue
* open_date
* closed_date
* payment_hist_string

### 3. Enquiry Data

Previous credit enquiries:

* enquiry_type
* enquiry_amt
* enquiry_date

---

# Approach

The project was executed through the following stages:

## Stage 1: Data Understanding

The JSON datasets were loaded and normalized into tabular format.

Performed:

* Shape analysis
* Missing value analysis
* Data type inspection
* Duplicate checks
* Target distribution analysis

---

## Stage 2: Exploratory Data Analysis

Studied:

### Accounts Dataset

* Number of accounts per borrower
* Credit type distribution
* Loan amount distribution
* Overdue amount distribution
* Account age distribution

### Enquiry Dataset

* Number of enquiries
* Enquiry amount distribution
* Enquiry recency

### Target Analysis

Investigated relationships between:

* Account counts
* Loan amounts
* Overdues
* Enquiries

and default behaviour.

---

# Feature Engineering

The most important part of the project.

All account-level and enquiry-level records were aggregated to borrower level using uid.

---

## A. Account Features

Created:

### Portfolio Size

* num_accounts
* num_open_accounts
* num_closed_accounts

### Loan Exposure

* total_loan_amount
* avg_loan_amount
* max_loan_amount

### Overdue Behaviour

* total_overdue
* avg_overdue
* max_overdue
* overdue_ratio

### Account Age

* oldest_account_age
* newest_account_age
* avg_account_age

### Credit Diversity

* num_credit_types
* consumer_credit_count
* credit_card_count
* mortgage_flag

---

## B. Payment History Features

The payment_hist_string field was parsed carefully.

Each 3-digit block represents Days Past Due (DPD).

Example:

000026

means:

* Previous month = 0 DPD
* Latest month = 26 DPD

The string was converted into numerical delinquency measures.

Created:

### Delinquency Features

* max_dpd_ever
* recent_dpd
* avg_dpd
* dpd_30_plus
* dpd_60_plus
* dpd_90_plus
* delinquency_months
* clean_payment_flag

These features captured both:

* Severity of delinquency
* Recency of delinquency

---

## C. Enquiry Features

Created:

### Volume Features

* num_enquiries

### Amount Features

* total_enquiry_amt
* avg_enquiry_amt
* max_enquiry_amt

### Recency Features

* recent_enquiry_days
* avg_enquiry_age

### Enquiry Diversity

* num_enquiry_types

---

# Data Preprocessing

Performed:

* Date conversion
* Feature aggregation
* Missing value treatment
* Median imputation
* Consistent train-test transformation

No target leakage was introduced.

---

# Modelling

Multiple models were explored.

## 1. Logistic Regression

Used as a baseline model.

Advantages:

* Interpretable
* Fast training
* Provides coefficient-based insights

---

## 2. Random Forest

Used to capture:

* Non-linear relationships
* Feature interactions

---

## 3. XGBoost

Used because:

* Handles tabular credit-risk data effectively
* Captures complex interactions
* Robust to feature scaling
* Generally performs strongly on structured datasets

Hyperparameter tuning was performed using cross-validation.

---

# Validation Strategy

Used:

* Stratified K-Fold Cross Validation

Reason:

* Preserves target distribution
* Produces more reliable ROC-AUC estimates

Evaluation Metric:

ROC-AUC

---

# Feature Importance Analysis

Feature importance was examined using:

* Logistic Regression coefficients
* Random Forest importance
* XGBoost importance

Key predictive features included:

* Delinquency measures
* Overdue ratios
* Recent DPD behaviour
* Number of accounts
* Enquiry activity

---

# Final Model

The final submission was generated using the best-performing model selected on validation ROC-AUC performance.

Predictions are submitted as probabilities rather than hard classifications.

---

# Files Included

## Notebook

Lakshya_Vipassana_Monsoon_CreditRisk.ipynb

Contains:

* Data Loading
* EDA
* Feature Engineering
* Model Training
* Validation
* Submission Generation

## Submission

final_submission_lakshya_vipassana.csv

Contains:

* uid
* probability prediction

---

# Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebook:

```bash
jupyter notebook
```

Execute all cells sequentially to reproduce results.

---

# Key Learning

The largest performance gains came from:

1. Borrower-level aggregation
2. Payment history parsing
3. Delinquency-based features
4. Enquiry recency features
5. Gradient boosting models

These features captured both historical repayment discipline and recent credit-seeking behaviour, which are critical indicators in credit risk modelling.
