.PHONY: help install train notebook clean

help:
	@echo "Targets:"
	@echo "  install   Install dependencies from requirements.txt"
	@echo "  train     Run the full training pipeline (scripts/train.py)"
	@echo "  notebook  Launch the analysis notebook"
	@echo "  clean     Remove generated artifacts (figures, metrics, model)"

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

notebook:
	jupyter notebook notebooks/easyvisa_approval_prediction.ipynb

clean:
	rm -rf reports/figures reports/metrics.json models/*.joblib
