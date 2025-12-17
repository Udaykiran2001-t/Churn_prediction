# Customer Churn Prediction Using Machine Learning

This repository contains an **end-to-end Machine Learning project** that predicts whether a bank customer will churn (leave the bank) using historical customer data. The project covers **EDA, feature engineering, preprocessing, model building, hyperparameter tuning, and evaluation**.

---

## 1. Project Overview

**Problem Statement:**
Customer churn is a major challenge for banks. Retaining existing customers is more cost-effective than acquiring new ones. This project aims to build a **machine learning model** that can predict customer churn and help banks take preventive actions.

**Objective:**

* Analyze customer behavior
* Identify key churn drivers
* Build and compare multiple ML models
* Select the best-performing model

---

## 2. Dataset Information

* **Dataset Name:** churn.csv
* **Records:** 10,000 customers
* **Features:** 13 original features
* **Target Variable:** `Exited`

  * 0 → Customer Stayed
  * 1 → Customer Churned

### Features

* CustomerId
* Surname
* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary
* Exited (Target)

---

## 3. Project Folder Structure

```
Customer-Churn-Prediction/
│
├── data/
│   └── churn.csv
│
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── images/
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Exploratory Data Analysis (EDA)

Key analysis performed:

* Class distribution of churned vs non-churned customers
* Churn analysis by:

  * Gender
  * Geography
  * Age groups
  * Balance
* Correlation analysis

### Key Insights

* Churn rate ≈ **20%** (imbalanced dataset)
* Older customers churn more
* Germany has the highest churn rate
* Inactive members are more likely to churn

---

## 5. Data Preprocessing

### 5.1 Missing & Outlier Analysis

* No missing values found
* No significant outliers (IQR method)

### 5.2 Feature Engineering

* Age and credit score binning using quantiles
* Monthly salary derived from annual salary
* New age-tenure interaction feature

### 5.3 Encoding

* One-hot encoding for categorical variables
* Dropped irrelevant identifiers (CustomerId, Surname)

### 5.4 Scaling

* RobustScaler used for numerical features
* Categorical variables kept unscaled

### 5.5 Handling Imbalanced Data

* **SMOTETomek** applied to balance churn classes

---

## 6. Model Building

The following models were trained and compared:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Gradient Boosting
* LightGBM
* CatBoost

### Model Performance (Accuracy)

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 74%      |
| KNN                 | 75%      |
| Decision Tree       | 79%      |
| Random Forest       | 85%      |
| Gradient Boosting   | 88%      |
| LightGBM            | 90%      |
| CatBoost            | 91%      |

---

## 7. Hyperparameter Tuning

* **GridSearchCV** used for tuning
* Tuned models:

  * Gradient Boosting
  * LightGBM
  * CatBoost

**Best Model:** LightGBM

---

## 8. Model Evaluation

### Metrics Used

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Visualizations

* Confusion Matrix
* Feature Importance
* ROC Curve

The LightGBM model achieved:

* High accuracy
* Strong ROC-AUC score
* Balanced precision and recall

---

## 9. Results & Business Impact

* Identified high-risk customers accurately
* Age, activity status, geography, and balance are key churn drivers
* Model can help banks design **retention strategies**

---

## 10. Installation & Usage

### Clone the Repository

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

* Open Jupyter Notebook
* Run notebooks in sequence from `notebooks/`

---

## 11. Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Imbalanced-learn

---

## 12. Future Improvements

* Deploy model using Flask / FastAPI
* Create a real-time churn prediction API
* Add SHAP explainability
* Improve recall for churned customers

---

## 13. Author

**Name:** Uday
**Domain:** Machine Learning / Data Science

---

⭐ If you like this project, don’t forget to give it a star on GitHub!
