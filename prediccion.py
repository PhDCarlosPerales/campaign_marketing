import pickle

import pandas as pd

with open("megamodelo.pkl", "rb") as f:
    clf = pickle.load(f)

df2 = pd.read_excel("ventas.xlsx")
df2.head()

y_predict = clf.predict(df2.drop("ventas", axis=1))

y_predict_proba = clf.predict_proba(df2.drop("ventas",axis=1))
y_predict_proba
