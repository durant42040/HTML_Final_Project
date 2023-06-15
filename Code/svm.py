import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, LinearSVC, SVR

start = time.time()
# data preprocessing
data = pd.read_csv('../training_data/train_encoded.csv')

X = data.drop('Danceability', axis=1)
y = data['Danceability']
sc = StandardScaler()


def svm():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalization
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    svm = SVR(cache_size=1000)
    svm.fit(X_train, y_train)

    Ein = mean_absolute_error(np.round(svm.predict(X_train)), np.array(y_train))
    Eout = mean_absolute_error(np.round(svm.predict(X_test)), np.array(y_test))

    print(f"Ein = {Ein}")
    print(f"Eout = {Eout}")

    test = pd.read_csv('../testing_data/test_encoded.csv')
    test = sc.transform(test)

    y_pred = svm.predict(X_test)
    return y_pred


danceability = np.zeros(3434)
N = 10

for i in range(N):
    danceability += svm() / N
np.round(danceability)

danceability_df = pd.DataFrame(np.round(danceability))
danceability_df.to_csv('results.csv', index=False, header=False)

print(danceability)
end = time.time()

print(f"time {end - start}")
