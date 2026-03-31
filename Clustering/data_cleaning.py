import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_cleaning(path):

    #reading data, semicolon separated
    df = pd.read_csv(path, sep=",")

    #removing row if too much missing columns
    df.dropna(thresh=len(df.columns)*0.3, inplace=True)

    #removing customer id column
    df = df.drop("CustomerID", axis=1)

    #missing values in numerical columns
    numerical_cols = df.select_dtypes(include="number").columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    #missing values in categorical columns
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

    #changing F/M columns with 0/1    
    df["Gender"] = df["Gender"].map({"M": 0, "F": 1})

    #standarization
    df = standarize(df, numerical_cols)

    return df

def standarize(df, cols):
    #standarization / standard scaling
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df