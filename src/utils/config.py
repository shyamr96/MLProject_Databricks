from pathlib import Path

import yaml


def load_config(path="config/params.yaml"):
	config_path = Path(path)

	# 1) Absolute path as provided.
	if config_path.is_absolute() and config_path.exists():
		with open(config_path, "r") as f:
			return yaml.safe_load(f)

	# 2) Path relative to current working directory.
	cwd_path = Path.cwd() / config_path
	if cwd_path.exists():
		with open(cwd_path, "r") as f:
			return yaml.safe_load(f)

	# 3) Path relative to repo root (two levels above src/utils).
	repo_path = Path(__file__).resolve().parents[2] / config_path
	if repo_path.exists():
		with open(repo_path, "r") as f:
			return yaml.safe_load(f)

	raise FileNotFoundError(f"Could not find config file at '{path}'")
