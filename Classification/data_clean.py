import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_cleaning(path):

    #reading data, semicolon separated
    df = pd.read_csv(path, sep=";")

    #standarization of all features except target
    feature_cols = df.columns.drop("quality")
    df = standarize(df, feature_cols)

    return df

def standarize(df, cols):
    #standarization / standard scaling
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def normalize(df, cols):
    #normalization / min-max scaling
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df