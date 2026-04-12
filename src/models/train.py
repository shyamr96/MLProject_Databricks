import os

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

MODEL_PATH = "/tmp/mlops_model/model.pkl"


def train_model(df, target):
	X = df.drop(columns=[target])
	y = df[target]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	model = LinearRegression()
	model.fit(X_train, y_train)

	os.makedirs("/tmp/mlops_model", exist_ok=True)
	joblib.dump(model, MODEL_PATH)

	return model
