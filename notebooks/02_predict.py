# COMMAND ----------
from src.utils.config import load_config
from src.models.predict import predict

# COMMAND ----------
config = load_config()

# COMMAND ----------
predict(
	data_path=config["data"]["predict_path"],
	output_path=config["data"]["output_path"]
)
