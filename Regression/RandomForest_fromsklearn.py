from data_clean import data_cleaning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import os
path = os.path.join(os.path.dirname(__file__), "studentperform.csv")

#data preprocessing
df, numerical_cols = data_cleaning(path)

#splitting data
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

#creating train and test X's and train and test y's
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#creating random forest model and fitting
model = RandomForestRegressor(
    n_estimators=100,    
    max_depth=20,        
    min_samples_split=2, 
    min_samples_leaf=3,  
    random_state=42
)

model.fit(X_train, y_train)

#predicting y values
y_pred = model.predict(X_test)

#evaluation of the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")

with open(os.path.join(os.path.dirname(__file__), "results.txt"), "a") as f:
    f.write(f"RandomForest_fromsklearn.py: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}\n")

