# COMMAND ----------
import pickle
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.utils.config import load_config
from src.data.ingestion import load_data
from src.models.train import train_model

spark = globals().get("spark")
if spark is None:
	raise RuntimeError("Spark session is not available. Run this notebook in Databricks.")

# COMMAND ----------
config = load_config()

# Set MLflow experiment
mlflow.set_experiment("/Users/rathodshyam2301@gmail.com/sales-prediction-experiment")

# COMMAND ----------
df = load_data(table_name=config["data"]["train_table"], spark=spark)

# COMMAND ----------
# Start MLflow run
with mlflow.start_run(run_name="sales-model-training"):
    # Log parameters
    mlflow.log_param("train_table", config["data"]["train_table"])
    mlflow.log_param("target_column", config["model"]["target_column"])
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("num_samples", len(df))
    mlflow.log_param("num_features", len(df.columns) - 1)
    
    # Train model
    model = train_model(df, config["model"]["target_column"])
    
    # Calculate metrics on test set
    X = df.drop(columns=[config["model"]["target_column"]])
    y = df[config["model"]["target_column"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate and log metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    
    # Log model to MLflow
    mlflow.sklearn.log_model(model, "sales-model")
    
    # Print model information
    print('Testing CI/CD Pipeline with MLflow')
    print("=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(f"Model Type: {type(model).__name__}")
    print(f"Model Coefficients: {model.coef_}")
    print(f"Model Intercept: {model.intercept_}")
    print(f"Number of Features: {len(model.coef_)}")
    print("\nModel Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print("=" * 60)
    print(f"\n✓ Experiment logged to MLflow")
    print(f"  Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------
# Save model to Unity Catalog for production use
model_blob = pickle.dumps(model)
spark.createDataFrame([(model_blob,)], ["model_blob"]).write.mode("overwrite").saveAsTable(
	config["model"]["model_table"]
)

print(f"\n✓ Model saved to Unity Catalog: {config['model']['model_table']}")
print("  (Production model for predictions)")