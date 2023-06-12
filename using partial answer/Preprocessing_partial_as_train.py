import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  PowerTransformer
from sklearn.compose import ColumnTransformer

# Load data
data = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
partial = pd.read_csv("test_partial_answer.csv")

# Merge the datasets based on the 'id' column
external_validation = pd.merge(partial, test_set, on='id')



# Move the 'id' column to the 18th column
columns = data.columns.tolist()
columns.insert(17, columns.pop(columns.index('id')))

data = pd.concat([data, external_validation], ignore_index=True)


# Split data into features and target variable
X_train = data.iloc[:, [i for i in range(1, 17)] + [23,26,27,28]].copy()
y_train = data.iloc[:, 0]
X_test = test_set.iloc[:, [i for i in range(0, 16)] + [22,25,26,27]].copy()



# Identify column types
numeric_cols = X_train.select_dtypes(include=np.number).columns
character_cols = X_train.select_dtypes(include='object').columns

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')
X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

character_imputer_frequent = SimpleImputer(strategy='most_frequent')
X_train[character_cols[0:3]] = character_imputer_frequent.fit_transform(X_train[character_cols[0:3]])
X_test[character_cols[0:3]] = character_imputer_frequent.transform(X_test[character_cols[0:3]])

character_imputer_constant = SimpleImputer(strategy='constant')
X_train[character_cols[3:6]] = character_imputer_constant.fit_transform(X_train[character_cols[3:6]])
X_test[character_cols[3:6]] = character_imputer_constant.transform(X_test[character_cols[3:6]])

# Define preprocessor
yeojohnson_transformer = PowerTransformer(method='yeo-johnson')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', yeojohnson_transformer, numeric_cols),
    ], remainder='passthrough'
)

# Fit the preprocessor on the test data
preprocessor.fit(X_test)

# Transform the training and test data
X_train_final = preprocessor.transform(X_train)
X_test_final = preprocessor.transform(X_test)