# COMMAND ----------
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
