import pandas as pd

df = df.drop(['customerID'], axis = 1)
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
df.fillna(df["TotalCharges"].mean())
df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})