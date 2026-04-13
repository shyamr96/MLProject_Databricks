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

print("=" * 60)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 60)
print(f"Model Type: {type(model).__name__}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print("=" * 60)

# COMMAND ----------
predictions_df = predict(
	data_table=config["data"]["predict_table"],
	spark=spark,
	model=model
)

# Print predictions
print("\n" + "=" * 60)
print("PREDICTIONS GENERATED")
print("=" * 60)
print(f"Number of predictions: {len(predictions_df)}")
print("\nFinal Predictions (showing all rows):")
print(predictions_df.to_string())
print("=" * 60)

# COMMAND ----------
spark.createDataFrame(predictions_df).write.mode("overwrite").saveAsTable(
	config["data"]["output_table"]
)

print(f"\n✓ Predictions saved successfully to: {config['data']['output_table']}")
