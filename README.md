# 🚗 EV Car Price Prediction using Ridge Regression

## 📌 Project Overview

This project predicts the **price of Electric Vehicles (EVs) in India** using **Ridge Regression**.

The dataset contains information about EV cars such as:

- Brand
- Model
- Range
- Power
- Battery
- Price

The project uses preprocessing techniques such as **One-Hot Encoding** for categorical data and **Standard Scaling** for numerical data. Different Ridge Regression `alpha` values are tested to evaluate model performance.

---

## 🎯 Objective

The main objective of this project is to:

- Analyze EV car data.
- Preprocess categorical and numerical features.
- Build a Ridge Regression model.
- Test different regularization values (`alpha`).
- Predict EV car prices.
- Evaluate the model using MAE, RMSE, and R² score.

---

## 📂 Dataset

The project uses an EV car dataset containing information about electric vehicles in India.

### Features Used

| Feature | Description |
|---|---|
| Brand | EV car manufacturer |
| Model | EV car model |
| Range | Driving range of the vehicle |
| Power | Vehicle power |
| Battery | Battery capacity |
| Price | Target variable representing the car price |

### Target Variable

**Price**

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

---

## 📚 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

## 🔄 Project Workflow

### 1. Import Libraries

Required Python libraries are imported for data handling, preprocessing, visualization, model building, and evaluation.

### 2. Load Dataset

The EV dataset is loaded using Pandas:

```python
df = pd.read_csv("ev_car_India_dataset.csv")
```

### 3. Explore the Dataset

The dataset is inspected using:

```python
df.head()
df.shape
df.isnull().sum()
```

This helps understand the dataset structure and identify missing values.

### 4. Separate Features and Target

The target column is:

```python
Price
```

The input features are separated from the target variable.

### 5. Identify Feature Types

Categorical features:

```python
categorical = ["Brand", "Model"]
```

Numerical features:

```python
numerical = ["Range", "Power", "Battery"]
```

### 6. Data Preprocessing

Categorical features are transformed using **One-Hot Encoding**.

Numerical features are standardized using **StandardScaler**.

```python
preprocessor = ColumnTransformer([
    ("Categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical),

    ("numerical",
     StandardScaler(),
     numerical)
])
```

### 7. Ridge Regression

Ridge Regression is used to predict EV prices.

The project tests different values of alpha:

```python
alphas = [0.01, 0.1, 1, 10, 100]
```

The model is implemented using a Scikit-learn Pipeline.

### 8. Train-Test Split

The dataset is divided into training and testing sets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)
```

- **80%** of the data is used for training.
- **20%** is used for testing.

### 9. Model Evaluation

The model is evaluated
