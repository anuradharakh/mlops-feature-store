PYTHON ?= python

.PHONY: setup format format-check lint test quality
.PHONY: ingestion preprocessing features feast split
.PHONY: smoke experiments report audit audit-strict
.PHONY: pipeline clean-runtime

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

format:
	ruff format src scripts tests feature_repo

format-check:
	ruff format --check src scripts tests feature_repo

lint:
	ruff check src scripts tests feature_repo

test:
	pytest -v

quality: format-check lint test

ingestion:
	$(PYTHON) scripts/run_ingestion.py

preprocessing:
	$(PYTHON) scripts/run_preprocessing.py

features:
	$(PYTHON) scripts/run_feature_v1.py
	$(PYTHON) scripts/run_feature_v2.py

feast:
	$(PYTHON) scripts/run_feast.py --reset

split:
	$(PYTHON) scripts/run_training_split.py

smoke:
	$(PYTHON) scripts/run_model_pipeline.py

experiments:
	$(PYTHON) scripts/run_experiments.py --reset

report:
	$(PYTHON) scripts/build_report.py

audit:
	$(PYTHON) scripts/verify_submission.py

audit-strict:
	$(PYTHON) scripts/verify_submission.py --require-git-tracked --require-source-portable

pipeline:
	$(PYTHON) scripts/run_pipeline.py

clean-runtime:
	rm -f mlflow.db mlflow.db-shm mlflow.db-wal mlflow.db-journal
	rm -rf artifacts/mlflow
	rm -f feature_repo/data/registry.db
	rm -f feature_repo/data/registry.db-shm
	rm -f feature_repo/data/registry.db-wal
	rm -f feature_repo/data/online_store.db
	rm -f feature_repo/data/online_store.db-shm
	rm -f feature_repo/data/online_store.db-wal
