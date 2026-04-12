import joblib
import pandas as pd

MODEL_PATH = "/dbfs/FileStore/model/model.pkl"


def load_model():
	return joblib.load(MODEL_PATH)


def predict(data_path, output_path):
	model = load_model()

	df = pd.read_csv(data_path)

	predictions = model.predict(df)

	df["prediction"] = predictions

	df.to_csv(output_path, index=False)

	return df
