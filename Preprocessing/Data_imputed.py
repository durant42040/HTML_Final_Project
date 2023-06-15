# Impute Album_type, Licensed, official_video with mode, with One-hot encode categorical variables
# Impute mean value for numeric value

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load data
data = pd.read_csv("../training_data/train.csv")
test_set = pd.read_csv("../testing_data/test.csv")

# Split data into features and target variable
X_train = data.iloc[:, [i for i in range(1, 17)] + [23]].copy()  # Make a copy of the DataFrame
y_train = data.iloc[:, 0]
X_test = test_set.iloc[:, [i for i in range(0, 16)] + [22]].copy()  # Make a copy of the DataFrame

# Impute missing values in X_train and X_test
numeric_cols = X_train.select_dtypes(include=np.number).columns
character_cols = X_train.select_dtypes(include='object').columns

numeric_imputer = SimpleImputer(strategy='mean')
X_train.loc[:, numeric_cols] = numeric_imputer.fit_transform(X_train.loc[:, numeric_cols])
X_test.loc[:, numeric_cols] = numeric_imputer.transform(X_test.loc[:, numeric_cols])

character_imputer = SimpleImputer(strategy='most_frequent')
X_train.loc[:, character_cols] = character_imputer.fit_transform(X_train.loc[:, character_cols])
X_test.loc[:, character_cols] = character_imputer.transform(X_test.loc[:, character_cols])

# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[character_cols]))
X_test_encoded = pd.DataFrame(encoder.transform(X_test[character_cols]))


# Concatenate encoded features with numeric features
X_train_final = pd.concat([X_train[numeric_cols], X_train_encoded], axis=1)
X_test_final = pd.concat([X_test[numeric_cols], X_test_encoded], axis=1)

X_train_final.to_csv('./train_imputed.csv', header=True)
X_test_final.to_csv('./test_imputed.csv', header=True)