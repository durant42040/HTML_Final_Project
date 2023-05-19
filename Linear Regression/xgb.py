import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

start = time.time()
# data preprocessing
data = pd.read_csv('../train/train_encoded.csv')
X = data.drop('Danceability', axis=1)
y = data['Danceability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalization
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'colsample_bytree': 0.3,
    'subsample': 0.8,
    'eval_metric': 'mae',
}

# Create DMatrices
d_train = xgb.DMatrix(X_train, label=y_train)
d_test = xgb.DMatrix(X_test, label=y_test)

# Train the model
regressor = xgb.train(params, d_train, num_boost_round=1000, evals=[(d_test, 'test')])

# regression

test = pd.read_csv('../test/test_encoded.csv')
# test = sc.transform(test)
test = xgb.DMatrix(test)

danceability = np.round(regressor.predict(test))

for i in range(len(danceability)):
    if danceability[i] > 9:
        danceability[i] = 9
    if danceability[i] < 0:
        danceability[i] = 0

print(danceability)


print(f"Ein = {mean_absolute_error(np.round(regressor.predict(d_train)), np.array(y_train))}")
print(f"Eout = {mean_absolute_error(np.round(regressor.predict(d_test)), np.array(y_test))}")

danceability_df = pd.DataFrame(danceability)
danceability_df.to_csv('results.csv', index=False, header=False)

end = time.time()

print(f"time {end - start}")
