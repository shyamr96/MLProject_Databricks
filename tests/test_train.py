import pandas as pd

from src.models.train import train_model


def test_training():
	df = pd.DataFrame({
		"feature1": [1, 2, 3],
		"feature2": [4, 5, 6],
		"sales": [10, 20, 30],
	})

	model = train_model(df, "sales")

	assert model is not None
