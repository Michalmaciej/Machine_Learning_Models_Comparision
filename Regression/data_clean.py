import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_cleaning(path):

    #reading data
    df = pd.read_csv(path)

    #removing row if too much missing columns
    df.dropna(thresh=len(df.columns)*0.3, inplace=True)

    #missing values in numerical columns
    numerical_cols = df.select_dtypes(include="number").columns.drop("Exam_Score")
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    #missing values in categorical columns
    categorical_cols = df.select_dtypes(exclude="number").columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    #changing yes/no columns to 0/1 columns
    binary_cols = ["Extracurricular_Activities", "Internet_Access", "Learning_Disabilities"]
    df[binary_cols] = df[binary_cols].replace({"Yes": 1, "No": 0})

    #one hot coding to data without values hierarchy
    df = pd.get_dummies(df, columns=["Gender", "School_Type"])
    false_cols = ["Gender_Male", "Gender_Female", "School_Type_Private", "School_Type_Public"]
    df[false_cols] = df[false_cols].replace({False: 0, True: 1})

    #distance_from_home label encoding
    distance_order = {"Near": 0, "Moderate": 1, "Far": 2}
    df["Distance_from_Home"] = df["Distance_from_Home"].map(distance_order)

    #peer influence label encoding
    peer_order = {"Negative": -1, "Neutral": 0, "Positive": 1}
    df["Peer_Influence"] = df["Peer_Influence"].map(peer_order)

    #parental education level label encoding
    parental_ed_order = {"High School": 0, "College": 1, "Postgraduate": 2}
    df["Parental_Education_Level"] = df["Parental_Education_Level"].map(parental_ed_order)

    #low medium high label encoding
    order = {"Low": 0, "Medium": 1, "High": 2}
    label_cols = ["Access_to_Resources", "Teacher_Quality", "Parental_Involvement", "Motivation_Level", "Family_Income"]
    for col in label_cols:
        df[col] = df[col].map(order)
    
    return df, numerical_cols

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
