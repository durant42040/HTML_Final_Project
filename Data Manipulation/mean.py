import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv('../train_pruned.csv')

# Create an instance of the SimpleImputer class with the strategy parameter set to 'mean'
imputer = SimpleImputer(strategy='mean')

imputer.fit(data)

# Transform the dataset by filling the missing values with the mean of the corresponding column
data_imputed = pd.DataFrame(imputer.transform(data), columns=data.columns)

# put data_imputed into csv file
data_imputed.to_csv('../train_imputed.csv', index=False)