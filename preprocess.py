def prep_data(df):

    df = df.assign(la=df["Length1"] * df["Length2"] / 2)

    X = df[["Length1", "Length2", "Height", "Width", "la"]].values
    y = df["Weight"].values

    return X, y