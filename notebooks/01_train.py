# COMMAND ----------
import pickle

from src.utils.config import load_config
from src.data.ingestion import load_data
from src.models.train import train_model

spark = globals().get("spark")
if spark is None:
	raise RuntimeError("Spark session is not available. Run this notebook in Databricks.")

# COMMAND ----------
config = load_config()

# COMMAND ----------
df = load_data(table_name=config["data"]["train_table"], spark=spark)

# COMMAND ----------
model = train_model(df, config["model"]["target_column"])

# Print model information
print('Testing CI/CD Pipeline - Version 1')
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)
print(f"Model Type: {type(model).__name__}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print(f"Number of Features: {len(model.coef_)}")
print("=" * 60)

# COMMAND ----------
model_blob = pickle.dumps(model)
spark.createDataFrame([(model_blob,)], ["model_blob"]).write.mode("overwrite").saveAsTable(
	config["model"]["model_table"]
)

print(f"\n✓ Model saved successfully to: {config['model']['model_table']}")
