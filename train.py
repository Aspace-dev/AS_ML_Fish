import pandas as pd
from sklearn.neural_network import MLPRegressor
from joblib import dump
from preprocess import prep_data

df = pd.read_csv("fish_participant.csv")

X, y = prep_data(df)

mlpr = MLPRegressor()
mlpr.fit(X, y)

dump(mlpr, "mlpr_reg.joblib")