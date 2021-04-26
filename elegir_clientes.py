import pickle
import random

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel("ventas.xlsx")
df.describe(include="all")
df.info()

target = "ventas"
columnas_categoricas = ["genero",
                        "localizacion",
                        "educacion"]
columnas_numericas = ["edad", "salario"]

X_train, X_test, y_train, y_test = train_test_split(
    df.drop([target], axis=1),
    df[target],
    test_size=0.2,
    random_state=42
)

numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = OneHotEncoder(drop="first")

preprocesador = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, columnas_numericas)
        , ('cat', categorical_transformer, columnas_categoricas)
    ]
)

pipeline = Pipeline(
    steps=[
        ("pre", preprocesador)
        , ("modelo", DecisionTreeClassifier())
    ]
)

pipeline.set_params(
    modelo__max_depth=5,
    modelo__max_leaf_nodes=5
)
pipeline.fit(X_train, y_train)

fig = plt.figure(figsize=(10, 10))
tree.plot_tree(
    pipeline.named_steps["modelo"],
    feature_names=columnas_numericas + list(
        pipeline.named_steps["pre"].named_transformers_["cat"] \
            .get_feature_names(columnas_categoricas)
    ),
    class_names=target,
    filled=True
)

y_test_random = random.choices(
    population=[True, False],
    weights=[y_train.mean(), 1 - y_train.mean()],
    k=y_test.shape[0]
)
sum(y_test_random) / (len(y_test_random))

# Mean Accuracy
print("Score Train:", pipeline.score(X_train, y_train))
print("Score Test:", pipeline.score(X_test, y_test))
print("Score Azar:", accuracy_score(y_test,y_test_random))







with open("megamodelo.pkl", 'wb') as f:
    pickle.dump(pipeline, f, pickle.HIGHEST_PROTOCOL)