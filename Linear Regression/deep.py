import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error

# Load the data into a pandas DataFrame
start = time.time()

data = pd.read_csv('../train/train_encoded.csv')
X = data.drop('Danceability', axis=1)
y = data['Danceability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)))
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)))
model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)))
model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)))
model.add(Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)))
model.add(Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)))
model.add(Dense(1, activation='linear'))


# Compile the model
model.compile(loss='mean_absolute_error', optimizer='Nadam', metrics=['mean_absolute_error'])


model.fit(X_train, y_train, epochs=100, batch_size=100, validation_data=(X_test, y_test))

_, Eout = model.evaluate(X_test, y_test)
_, Ein = model.evaluate(X_train, y_train)

print(f'Eout: {Eout}')
print(f'Ein: {Ein}')

test = pd.read_csv('../test/test_encoded.csv')
test = scaler.transform(test)

danceability = model.predict(test)

danceability_df = pd.DataFrame(np.round(danceability))
# danceability_df.to_csv('results.csv', index=False)

end = time.time()
print(f'time: {end-start}s')