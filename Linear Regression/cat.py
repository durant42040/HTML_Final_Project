import catboost as cb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Assume you have a dataset with features 'X' and target variable 'y'
# Split the data into train and test sets
data = pd.read_csv('../train/train_encoded.csv')
X = data.drop('Danceability', axis=1)
y = data['Danceability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Album_type', 'Licensed', 'official_video', 'Composer', 'Artist', 'Title', 'Channel', 'Album']


model = cb.CatBoostRegressor(iterations=1000, depth=5, learning_rate=0.1, loss_function='MAE', eval_metric='MAE')

# Fit the model on the training data
model.fit(X_train, y_train, cat_features=categorical_features)

# Make predictions on the test data

# Evaluate the model
Eout = mean_absolute_error(y_test, model.predict(X_test))
Ein = mean_absolute_error(y_train, model.predict(X_train))
print("Eout:", Eout)
print("Ein:", Ein)

test = pd.read_csv('../test/test_encoded.csv')
danceability = model.predict(test)
for i in range(len(danceability)):
    if danceability[i] > 9:
        danceability[i] = 9
    if danceability[i] < 0:
        danceability[i] = 0

danceability = np.round(danceability)
print(danceability)
danceability_df = pd.DataFrame(danceability)
danceability_df.to_csv('results.csv', index=False)