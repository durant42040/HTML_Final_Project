import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('../train/train.csv')
test = pd.read_csv('../test/test.csv')
all_data = pd.concat([data, test])


le = LabelEncoder()

for i in all_data:
    if i != 'Danceability':
        all_data[i] = le.fit_transform(all_data[i])

data = all_data[:17170]
test = all_data[17170:]
test.drop('Danceability', axis=1)

data.to_csv('../train/train_.csv', index=False)
test.to_csv('../test/test_.csv', index=False)
