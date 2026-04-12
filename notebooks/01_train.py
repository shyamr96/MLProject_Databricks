# COMMAND ----------
from src.utils.config import load_config
from src.data.ingestion import load_data
from src.models.train import train_model

# COMMAND ----------
config = load_config()

# COMMAND ----------
df = load_data(config["data"]["train_path"])

# COMMAND ----------
model = train_model(df, config["model"]["target_column"])
