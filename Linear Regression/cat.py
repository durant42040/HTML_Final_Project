import time
import catboost as cb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

start = time.time()

# Split the data into train and test sets
data = pd.read_csv('../train/train_encoded.csv')
X = data.drop('Danceability', axis=1)
y = data['Danceability']
errors = []
categorical_features = ['Album_type', 'Licensed', 'official_video', 'Composer', 'Artist', 'Title', 'Channel', 'Album']


def catboost():
    global errors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # Normalization
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    model = cb.CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='MAE', eval_metric='MAE',
                                 l2_leaf_reg=5)
    # Fit the model on the training data
    model.fit(X_train, y_train, cat_features=categorical_features)

    # Make predictions on the test data

    # Evaluate the model
    Eout = mean_absolute_error(y_test, model.predict(X_test))
    Ein = mean_absolute_error(y_train, model.predict(X_train))
    print("Eout:", Eout)
    print("Ein:", Ein)
    errors.append(Eout)

    test = pd.read_csv('../test/test_encoded.csv')
    danceability = model.predict(test)

    for i in range(len(danceability)):
        if danceability[i] > 9:
            danceability[i] = 9
        if danceability[i] < 0:
            danceability[i] = 0

    return danceability


mean = np.zeros(6315)
for i in range(10):
    mean += catboost() / 10


print(errors)
print(np.mean(errors))
print(np.round(mean))

danceability_df = pd.DataFrame(np.round(mean))
danceability_df.to_csv('results.csv', index=False)

end = time.time()

print(f"time {end - start}")
