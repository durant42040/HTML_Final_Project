import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

start = time.time()
# data preprocessing
data = pd.read_csv('../training_data/train.csv')

X = data.drop('Danceability', axis=1)
y = data['Danceability']

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'colsample_bytree': 0.3,
    'subsample': 0.8,
    'eval_metric': 'mae',
    'gamma': 5
}

errors = []


def xgblinear():
    global errors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # Normalization
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test, label=y_test)

    # Train the model
    regressor = xgb.train(params, d_train, num_boost_round=1000, evals=[(d_test, 'testing_data')])

    Ein = mean_absolute_error(np.round(regressor.predict(d_train)), np.array(y_train))
    Eout = mean_absolute_error(np.round(regressor.predict(d_test)), np.array(y_test))
    print(f"Ein = {Ein}")
    print(f"Eout = {Eout}")
    errors.append(Eout)

    test = pd.read_csv('../testing_data/test_encoded.csv')
    # testing_data = sc.transform(testing_data)
    test = xgb.DMatrix(test)

    danceability = regressor.predict(test)
    for i in range(len(danceability)):
        if danceability[i] > 9:
            danceability[i] = 9
        if danceability[i] < 0:
            danceability[i] = 0

    return danceability


mean = np.zeros(6315)
for i in range(10):
    mean += xgblinear() / 10

print(errors)
print(np.mean(errors))
print(np.round(mean))

danceability_df = pd.DataFrame(mean)
danceability_df.to_csv('results2.csv', index=False)

end = time.time()

print(f"time {end - start}")
