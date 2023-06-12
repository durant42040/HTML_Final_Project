from catboost import CatBoostRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import numpy as np

categorical_features = [14, 15, 16,17,18,19]

# Define pipeline for CatBoost Regression
pipe_cat = make_pipeline(
    CatBoostRegressor(task_type='GPU', devices='0', random_state=42, verbose=0, cat_features=categorical_features)
)

# Define hyperparameter grid
param_grid_cat = {
    'catboostregressor__iterations': [500,1000],
    'catboostregressor__depth': [6,10],
    'catboostregressor__learning_rate': [0.1,0.5],
    'catboostregressor__l2_leaf_reg': [25,50,75]
}

# Make scorer objects for MSE and MAE
scoring = {'mse': make_scorer(mean_squared_error, greater_is_better=False),
           'mae': make_scorer(mean_absolute_error, greater_is_better=False)}

# Perform grid search with 5-fold cross-validation
grid_cat = GridSearchCV(pipe_cat, param_grid_cat, cv=5, scoring=scoring, refit='mse')
grid_cat.fit(X_train_final, y_train)

# Print the best parameters and cross-validation MSE and MAE
print("Best parameters:", grid_cat.best_params_)
print("Cross-validation MSE:", abs(grid_cat.cv_results_['mean_test_mse'][grid_cat.best_index_]))
print("Cross-validation MAE:", abs(grid_cat.cv_results_['mean_test_mae'][grid_cat.best_index_]))

# Train the model on the full training set with the best hyperparameters
model = CatBoostRegressor(task_type='GPU',
                          devices='0',
                          iterations=grid_cat.best_params_['catboostregressor__iterations'],
                          depth=grid_cat.best_params_['catboostregressor__depth'],
                          learning_rate=grid_cat.best_params_['catboostregressor__learning_rate'],
                          l2_leaf_reg = grid_cat.best_params_['catboostregressor__l2_leaf_reg'],
                          random_state=42,
                          cat_features=categorical_features,
                          verbose=0)

model.fit(X_train_final, y_train)

# Make predictions on the test set
cat_pred = model.predict(X_test_final)

# Round the predictions to the nearest integer
rounded_pred_cat = np.round(cat_pred).astype(int)

# Clip the predictions to be within the range of 0-9
clipped_pred_cat = np.clip(rounded_pred_cat, 0, 9)

# Update sample submission file
sample.iloc[:, 1] = clipped_pred_cat
sample.to_csv('sample_cat.csv', index=False)
