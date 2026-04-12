import pandas as pd


def load_data(path=None, table_name=None, spark=None):
	if table_name:
		if spark is None:
			raise ValueError("spark session is required when reading from a table")
		return spark.table(table_name).toPandas()

	if path:
		return pd.read_csv(path)

	raise ValueError("either path or table_name must be provided")
