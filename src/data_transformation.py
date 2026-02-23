from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def transform_data(df):

    # Target encoding
    le = LabelEncoder()
    df["RainTomorrow"] = le.fit_transform(df["RainTomorrow"])

    # Binary column
    df["RainToday"] = df["RainToday"].map({"No":0,"Yes":1})

    # Encode Location
    loc_encoder = LabelEncoder()
    df["Location"] = loc_encoder.fit_transform(df["Location"])

    # Encode wind direction
    cols = ["WindGustDir","WindDir9am","WindDir3pm"]

    encoder = OrdinalEncoder()
    df[cols] = encoder.fit_transform(df[cols])

    return df