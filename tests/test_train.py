import pandas as pd

from src.models.train import train_model


def test_training():
	# Use at least 10 rows for train_test_split to work properly
	df = pd.DataFrame({
		"feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		"feature2": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
		"sales": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
	})

	model = train_model(df, "sales")

	assert model is not None
	assert hasattr(model, "coef_")  # Verify it's a trained model
