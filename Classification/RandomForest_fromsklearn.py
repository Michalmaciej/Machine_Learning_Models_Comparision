from data_clean import data_cleaning
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import os
path = os.path.join(os.path.dirname(__file__), "winequality-white.csv")

#data preprocessing
df = data_cleaning(path)

#splitting data
X = df.drop("quality", axis=1)
y = df["quality"]

#creating train and test X's and train and test y's
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#creating random forest model and fitting
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

#predicting y values
y_pred = model.predict(X_test)

#evaluation of the model
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, average="weighted")
prec = precision_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy:  {acc:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"Precision: {prec:.2f}")
print(f"F1:        {f1:.2f}")

with open(os.path.join(os.path.dirname(__file__), "results.txt"), "a") as f:
    f.write(f"RandomForest_fromsklearn.py: Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}\n")