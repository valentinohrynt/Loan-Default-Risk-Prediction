# Loan Default Risk Prediction

A machine learning project to predict loan default risk using RandomForest classification.

## Overview

This project demonstrates a complete machine learning pipeline for predicting loan default risk, including:
- Data acquisition and preprocessing
- Exploratory data analysis and visualization
- Model training and evaluation
- Cross-validation for model robustness

## Dataset

**Source:** Loan Default Risk Prediction Dataset (Kaggle)

**Features:**
- `Retirement_Age`: Age of the individual at retirement
- `Debt_Amount`: Total debt amount
- `Monthly_Savings`: Monthly savings amount
- `Loan_Default_Risk`: Target variable (0 = No Default, 1 = Default)

## Getting Started

### Prerequisites

- Python 3.10+
- Kaggle API credentials properly configured

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/valentinohrynt/loan-default-risk-prediction.git
cd loan-default-risk-prediction
pip install -r requirements.txt
```

## Project Workflow

### 1. Data Loading

```python
import kagglehub
import pandas as pd

# Download dataset
path = kagglehub.dataset_download("himelsarder/loan-default-risk-prediction-dataset")

# Load into DataFrame
data = pd.read_csv(path + "/loan_default_risk_dataset.csv")
```

### 2. Data Cleaning

```python
# Remove missing values and duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
```

### 3. Exploratory Data Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot loan default distribution
plt.figure(figsize=(10, 6))
data["Loan_Default_Risk"].value_counts().plot(kind="bar", color=["salmon", "lightblue"])
plt.title("Loan Default Risk Distribution")
plt.show()
```

### 4. Feature Scaling & Data Splitting

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.drop("Loan_Default_Risk", axis=1)
y = data["Loan_Default_Risk"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### 5. Model Training

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### 6. Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = rf_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

### 7. Feature Importance Analysis

```python
feature_importances = pd.DataFrame({
    'Feature': ['Retirement_Age', 'Debt_Amount', 'Monthly_Savings'],
    'Importance': rf_model.feature_importances_
})

print(feature_importances.sort_values(by='Importance', ascending=False))
```

### 8. Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')

print(f"Cross Validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")
```

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 95% |
| Cross-Validation Mean Accuracy | 94.62% |
| Top Features | Monthly Savings, Debt Amount, Retirement Age |

## Project Structure

```
.
├── main.py                # Main script with all steps
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## How to Run

```bash
# Make sure Kaggle API credentials are set up
python main.py
```

## Author

This project is created to demonstrate a Machine Learning pipeline for educational purposes.

## License

This project is licensed under the MIT License.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/himelsarder/loan-default-risk-prediction-dataset)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Seaborn Documentation](https://seaborn.pydata.org/documentation.html)