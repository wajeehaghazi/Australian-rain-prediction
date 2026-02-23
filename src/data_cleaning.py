def clean_data(df):

    num_cols = df.select_dtypes(include=["int64","float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df