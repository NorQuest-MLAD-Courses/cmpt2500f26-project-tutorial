import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df = df.drop(['customerID'], axis = 1)

df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)

df.fillna(df["TotalCharges"].mean())

df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

df = df.apply(lambda x: object_to_int(x))
