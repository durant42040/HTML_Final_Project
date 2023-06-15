import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Suppress unnecessary pandas warnings
pd.options.mode.chained_assignment = None


def load_data():
    data = pd.read_csv('../training_data/train_encoded.csv')
    test = pd.read_csv('../testing_data/test_encoded.csv')
    private_data = pd.read_csv('../test_partial_answer.csv')
    return data, test, private_data


def preprocess_data(data, test):
    X = data.drop('Danceability', axis=1)
    y = data['Danceability']

    return X, y, test


def train_model(X_train, y_train, X_test, y_test):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'boosting_type': 'dart',
        'objective': 'regression',
        'num_leaves': 100,
        'bagging_freq': 1,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'metric': 'mae',
        'num_trees': 1000,
        'verbose': 0
    }

    rf_regressor = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval])
    return rf_regressor


def evaluate_model(model, X_train, y_train, X_test, y_test, X_private, y_private):
    Ein = mean_absolute_error(y_train, model.predict(X_train))
    Eval = mean_absolute_error(y_test, model.predict(X_test))
    Eout = mean_absolute_error(y_private, model.predict(X_private))

    print("Ein:", Ein)
    print("Eval:", Eval)
    print('Eout:', Eout)


def make_submission(model, test, ids):
    danceability = np.round(model.predict(test))
    submission = pd.DataFrame({'id': ids, 'Danceability': danceability})
    submission.to_csv('sample_dart.csv', index=False)


def main():
    start = time.time()

    data, test, private_data = load_data()
    X, y, test = preprocess_data(data, test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_private = test.iloc[10000:]
    for id in private_data['id'].to_list():
        X_private.loc[len(X_private)] = test.loc[id-17170]
    y_private = private_data['Danceability']

    model = train_model(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_train, y_train, X_test, y_test, X_private, y_private)
    make_submission(model, test, range(17170, 23485))

    end = time.time()
    print(f'Total execution time: {end - start}s')


if __name__ == '__main__':
    main()