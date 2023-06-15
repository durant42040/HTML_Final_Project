import time
import catboost as cb
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

start = time.time()

data = pd.read_csv('../training_data/train.csv')
test = pd.read_csv('../testing_data/test.csv')

X = data.drop(['Danceability'], axis=1)
y = data['Danceability']

combined_data = pd.concat([X, test], ignore_index=True)
combined_data = combined_data.drop(['Description', 'Uri', 'Url_spotify', 'Url_youtube', 'Track', 'id'], axis=1)

# Impute categorical values
categorical_cols = ['Key', 'Album_type', 'Licensed', 'official_video', 'Album', 'Channel', 'Composer', 'Artist', 'Title']

imputer = SimpleImputer(strategy='most_frequent')
impute_cols = ['Album_type', 'Licensed', 'official_video', 'Key']
combined_data[impute_cols] = imputer.fit_transform(combined_data[impute_cols])
character_imputer_frequent = SimpleImputer(strategy='most_frequent')


combined_data['Artist'] = combined_data['Title'].str.split('-').str[0]

label_encoder = LabelEncoder()
columns = combined_data.columns.tolist()
for feature in columns:
    combined_data[feature] = label_encoder.fit_transform(combined_data[feature])

X = combined_data.loc[:len(X) - 1]

test = combined_data.loc[len(X):]

errors = []


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
    model.fit(X_train, y_train, cat_features=categorical_cols, eval_set=(X_test, y_test))

    # Make predictions on the testing_data data

    # Evaluate the model
    Eout = mean_absolute_error(y_test, model.predict(X_test))
    Ein = mean_absolute_error(y_train, model.predict(X_train))
    print("Eout:", Eout)
    print("Ein:", Ein)
    errors.append(Eout)

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

submission = pd.DataFrame({'id': range(17170, 23485), 'Danceability': np.round(mean)})
submission.to_csv('submission.csv', index=False)

end = time.time()

print(f"time {end - start}")
