# COMMAND ----------
import pickle

from src.utils.config import load_config
from src.models.predict import predict

spark = globals().get("spark")
if spark is None:
	raise RuntimeError("Spark session is not available. Run this notebook in Databricks.")

# COMMAND ----------
config = load_config()

# COMMAND ----------
model_row = spark.table(config["model"]["model_table"]).select("model_blob").first()
if model_row is None:
	raise RuntimeError("No trained model found. Run notebooks/01_train.py first.")

model = pickle.loads(model_row["model_blob"])

# COMMAND ----------
predictions_df = predict(
	data_table=config["data"]["predict_table"],
	spark=spark,
	model=model
)

# COMMAND ----------
spark.createDataFrame(predictions_df).write.mode("overwrite").saveAsTable(
	config["data"]["output_table"]
)
