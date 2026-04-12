# MLOps Pipeline on Databricks

An end-to-end MLOps pipeline that trains a Linear Regression model on Unity Catalog table data and writes predictions back to Unity Catalog — all running inside Databricks Repos with no dependency on DBFS.

---

## Project Structure

```
mlops-databricks/
├── config/
│   └── params.yaml          # Central config: table names, target column
├── data/
│   ├── train.csv            # Sample training data (local reference)
│   └── new_data.csv         # Sample prediction input data (local reference)
├── notebooks/
│   ├── 01_train.py          # Databricks notebook: trains and saves model
│   └── 02_predict.py        # Databricks notebook: loads model, runs predictions
├── src/
│   ├── data/
│   │   └── ingestion.py     # Data loading from CSV or Unity Catalog table
│   ├── models/
│   │   ├── train.py         # Model training logic (LinearRegression)
│   │   └── predict.py       # Model prediction logic
│   └── utils/
│       └── config.py        # YAML config loader (Databricks Repos aware)
├── tests/
│   └── test_train.py        # Unit test for training function
└── requirements.txt         # Python dependencies
```

---

## Configuration

All runtime paths and table names are centralised in [config/params.yaml](config/params.yaml):

```yaml
data:
  train_table: "workspace.practice_schema.train"
  predict_table: "workspace.practice_schema.new_data"
  output_table: "workspace.practice_schema.predictions"

model:
  model_table: "workspace.practice_schema.sales_model_artifact"
  target_column: sales
```

| Key | Description |
|-----|-------------|
| `data.train_table` | Unity Catalog table used as training input |
| `data.predict_table` | Unity Catalog table used as prediction input (must NOT contain target column) |
| `data.output_table` | Unity Catalog table where predictions are written |
| `model.model_table` | Unity Catalog table where serialized model artifact is stored |
| `model.target_column` | Name of the target column in the training table |

---

## Unity Catalog Tables

### Prerequisites

The following tables must exist in your Unity Catalog before running:

| Table | Columns | Notes |
|-------|---------|-------|
| `workspace.practice_schema.train` | `feature1`, `feature2`, `sales` | Training data with target column |
| `workspace.practice_schema.new_data` | `feature1`, `feature2` | Prediction input — must NOT contain `sales` |

The following tables are created automatically by the pipeline:

| Table | Created by | Description |
|-------|-----------|-------------|
| `workspace.practice_schema.sales_model_artifact` | `notebooks/01_train.py` | Serialized trained model stored as binary blob |
| `workspace.practice_schema.predictions` | `notebooks/02_predict.py` | Prediction results with `prediction` column appended |

---

## Pipeline Flow

### Step 1 — Training (`notebooks/01_train.py`)

```
Unity Catalog (train table)
        │
        ▼
  load_data()          ← reads with spark.table().toPandas()
        │
        ▼
  train_model()        ← LinearRegression, train/test split 80/20
        │
        ▼
  pickle.dumps(model)  ← serializes sklearn model to bytes
        │
        ▼
Unity Catalog (sales_model_artifact table)
```

1. Loads active Spark session from Databricks runtime globals.
2. Reads `config/params.yaml` (resolves path relative to repo root automatically).
3. Reads training data from `workspace.practice_schema.train` via `spark.table()`.
4. Trains `LinearRegression` model using `sklearn`.
5. Serializes trained model with `pickle` and stores it as a one-row binary table in `workspace.practice_schema.sales_model_artifact`.

### Step 2 — Prediction (`notebooks/02_predict.py`)

```
Unity Catalog (sales_model_artifact table)
        │
        ▼
  pickle.loads(blob)   ← deserializes model from catalog
        │
        ▼
Unity Catalog (new_data table)
        │
        ▼
  predict()            ← model.predict() on feature columns
        │
        ▼
Unity Catalog (predictions table)
```

1. Loads Spark session.
2. Reads config.
3. Loads serialized model bytes from `workspace.practice_schema.sales_model_artifact`.
4. Deserializes model with `pickle`.
5. Reads prediction input from `workspace.practice_schema.new_data`.
6. Runs `model.predict()` on the input.
7. Appends `prediction` column to the DataFrame.
8. Writes results to `workspace.practice_schema.predictions` (overwrite mode).

---

## How to Run in Databricks

### 1. Clone the Repo in Databricks

In Databricks:
1. Go to **Repos** → **Add Repo**.
2. Enter: `https://github.com/shyamr96/MLProject_Databricks.git`
3. Click **Create Repo**.

Your repo will be available at:
```
/Workspace/Repos/<your-email>/MLProject_Databricks
```

### 2. Prepare Input Tables

Ensure `workspace.practice_schema.train` and `workspace.practice_schema.new_data` exist with correct columns.

To drop the target column from prediction input if it was accidentally included:

```sql
CREATE OR REPLACE TABLE workspace.practice_schema.new_data AS
SELECT feature1, feature2
FROM workspace.practice_schema.new_data;
```

### 3. Run Training Notebook

Open and run **`notebooks/01_train.py`** in Databricks.

Expected output per cell:
- Cell 1: imports and Spark session guard
- Cell 2: config loaded
- Cell 3: training DataFrame with `feature1`, `feature2`, `sales`
- Cell 4: trained model object
- Cell 5: model artifact written to `workspace.practice_schema.sales_model_artifact`

### 4. Run Prediction Notebook

Open and run **`notebooks/02_predict.py`** in Databricks.

Expected output per cell:
- Cell 1: imports and Spark session guard
- Cell 2: config loaded
- Cell 3: model loaded from catalog
- Cell 4: predictions DataFrame with `prediction` column
- Cell 5: results written to `workspace.practice_schema.predictions`

### 5. Verify Predictions

Run in a SQL cell or Databricks SQL:

```sql
SELECT * FROM workspace.practice_schema.predictions LIMIT 10;
```

---

## Module Details

### `src/utils/config.py`

Loads `config/params.yaml` using a three-step path resolution strategy:

1. Absolute path (if provided).
2. Path relative to current working directory.
3. Path relative to repo root (resolved from module file location) — handles Databricks Repos working directory.

### `src/data/ingestion.py`

```python
load_data(path=None, table_name=None, spark=None)
```

- If `table_name` is provided: reads from Unity Catalog via `spark.table().toPandas()`.
- If `path` is provided: reads from CSV file.

### `src/models/train.py`

```python
train_model(df, target)
```

- Splits features and target.
- Runs 80/20 train/test split.
- Fits `LinearRegression`.
- Returns the trained model.

### `src/models/predict.py`

```python
predict(data_path=None, output_path=None, data_table=None, spark=None, model=None)
```

- Accepts a preloaded `model` (used in Databricks to pass catalog-loaded model).
- Falls back to loading from local file if no model is passed.
- Reads input from table or CSV.
- Returns DataFrame with `prediction` column appended.

---

## Dependencies

```
pandas
scikit-learn
pyyaml
joblib
pytest
```

All are pre-installed on Databricks runtime clusters.

---

## Running Tests Locally

> Note: Tests are designed for local validation only. The Databricks pipeline runs directly in notebooks.

```bash
pytest -q
```

The unit test in `tests/test_train.py` validates the `train_model()` function with a small in-memory DataFrame.

---

## Important Notes

- **Run order matters**: Always run `01_train.py` before `02_predict.py`. The prediction notebook will raise a clear `RuntimeError` if no model artifact table is found.
- **Prediction input must not contain target column**: Remove `sales` from `new_data` table before running predictions.
- **Model persistence**: The trained model is stored as a binary blob in Unity Catalog — not in `/tmp` — so it survives cluster restarts and separate job runs.
