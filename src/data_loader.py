import pandas as pd

def load_data():
    df = pd.read_csv("data/weatherAUS.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df