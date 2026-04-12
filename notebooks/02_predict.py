# COMMAND ----------
from src.utils.config import load_config
from src.models.predict import predict

spark = globals().get("spark")
if spark is None:
	raise RuntimeError("Spark session is not available. Run this notebook in Databricks.")

# COMMAND ----------
config = load_config()

# COMMAND ----------
predictions_df = predict(
	data_table=config["data"]["predict_table"],
	spark=spark
)

# COMMAND ----------
spark.createDataFrame(predictions_df).write.mode("overwrite").saveAsTable(
	config["data"]["output_table"]
)
