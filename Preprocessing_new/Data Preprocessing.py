# Impute Album_type, Licensed, official_video with mode, with One-hot encode categorical variables
# Impute mean value for numeric value, with quantile transform
# Add noise to training data

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


# Load data
data = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")


# Split data into features and target variable
X_train = data.iloc[:, [i for i in range(1, 17)] + [23]].copy()
y_train = data.iloc[:, 0]
X_test = test_set.iloc[:, [i for i in range(0, 16)] + [22]].copy()

# Identify column types
numeric_cols = X_train.select_dtypes(include=np.number).columns
character_cols = X_train.select_dtypes(include='object').columns

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')
X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

character_imputer = SimpleImputer(strategy='most_frequent')
X_train[character_cols] = character_imputer.fit_transform(X_train[character_cols])
X_test[character_cols] = character_imputer.transform(X_test[character_cols])

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', QuantileTransformer(n_quantiles=500, output_distribution='normal'), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), character_cols)
    ])

# Fit and transform the training data
X_train_final = preprocessor.fit_transform(X_train)

# Transform the test data
X_test_final = preprocessor.transform(X_test)

# Add Gaussian noise to the numerical features of the training set
sigma = 0.1
X_train_final[:, :len(numeric_cols)] = X_train_final[:, :len(numeric_cols)] + np.random.normal(0, sigma, X_train_final[:, :len(numeric_cols)].shape)

# Convert to DataFrame
X_test_final_df = pd.DataFrame(X_test_final)
X_train_final_df = pd.DataFrame(X_train_final)
# Plot histograms
X_test_final_df.hist(bins=50, figsize=(20,15))
plt.show()
X_train_final_df.hist(bins=50, figsize=(20,15))
plt.show()
