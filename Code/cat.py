import numpy as np
import time
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import catboost as cb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

start = time.time()

# Load your dataset
data = pd.read_csv('../train/train.csv')

test = pd.read_csv('../test/test.csv')
X = data.drop('Danceability', axis=1)
y = data['Danceability']
combined_data = pd.concat([X, test], ignore_index=True)
combined_data = combined_data.drop(['Uri'], axis=1)
# ['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness',
#        'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms',
#        'Views', 'Likes', 'Stream', 'Album_type', 'Licensed', 'official_video',
#        'id', 'Track', 'Album', 'Url_spotify', 'Url_youtube', 'Comments',
#        'Description', 'Title', 'Channel', 'Composer', 'Artist']


# impute categorical values
categorical_cols = list(np.array(combined_data.select_dtypes(include='object').columns))
imputer = SimpleImputer(strategy='most_frequent')
combined_data[categorical_cols] = imputer.fit_transform(combined_data[categorical_cols])

# impute numerical values
numeric_cols = list(combined_data.select_dtypes(include=np.number).columns)
numeric_cols.remove('Key')
# imputer = IterativeImputer(estimator=LinearRegression())
imputer = SimpleImputer(strategy='mean')
# sc = StandardScaler()
# combined_data[numeric_cols] = sc.fit_transform(combined_data[numeric_cols])
combined_data[numeric_cols] = imputer.fit_transform(combined_data[numeric_cols])

label_encoder = LabelEncoder()
combined_data['Key'] = label_encoder.fit_transform(combined_data['Key'])

X = combined_data.loc[:len(X)-1]
test = combined_data.loc[len(X):]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = cb.CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='MAE',
                                 eval_metric='MAE', l2_leaf_reg=7, use_best_model=True)
# Fit the model on the training data
model.fit(X_train, y_train, cat_features=categorical_cols, eval_set=(X_test, y_test))

# Make predictions on the test data

# Evaluate the model
Eout = mean_absolute_error(y_test, model.predict(X_test))
Ein = mean_absolute_error(y_train, model.predict(X_train))
print("Eout:", Eout)
print("Ein:", Ein)

danceability = np.round(model.predict(test))
print(danceability)
submission = pd.DataFrame({'id': test['id'], 'danceability': danceability})
submission.to_csv('submission.csv', index=False)

end = time.time()
print(f'time: {end - start}s')
