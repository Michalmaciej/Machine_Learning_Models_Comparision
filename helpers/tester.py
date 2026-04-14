from train_test_split import train_test_split
import numpy as np
import pandas as pd

#numpy arrays
X_np = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                  [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
y_np = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

#dataframes
X_df = pd.DataFrame({"a": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                     "b": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]})
y_df = pd.DataFrame({"label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

#lists
X_list = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
          [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
y_list = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

#testing numpy arrays
print("=== NUMPY ===")
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train: {y_train}, y_test: {y_test}")

#testing dataframes
print("\n=== DATAFRAME ===")
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train:\n{y_train}\ny_test:\n{y_test}")

#testing lists
print("\n=== LIST ===")
X_train, X_test, y_train, y_test = train_test_split(X_list, y_list, test_size=0.3, random_state=42)
print(f"X_train len: {len(X_train)}, X_test len: {len(X_test)}")
print(f"y_train: {y_train}, y_test: {y_test}")

print(1)
print(2)
print(3)

