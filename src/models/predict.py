import joblib
import pandas as pd

MODEL_PATH = "/tmp/mlops_model/model.pkl"


def load_model():
	return joblib.load(MODEL_PATH)


def predict(data_path=None, output_path=None, data_table=None, spark=None):
	model = load_model()

	if data_table:
		if spark is None:
			raise ValueError("spark session is required when reading from a table")
		df = spark.table(data_table).toPandas()
	elif data_path:
		df = pd.read_csv(data_path)
	else:
		raise ValueError("either data_path or data_table must be provided")

	predictions = model.predict(df)

	df["prediction"] = predictions

	if output_path:
		df.to_csv(output_path, index=False)

	return df
