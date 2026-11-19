PY ?= python

.PHONY: all install download pipeline test backtest export notebook readme clean
all: install
	$(PY) run_all.py

install:
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) -m src.pipeline.download

pipeline:
	$(PY) -m src.pipeline.run

test:
	$(PY) -m pytest -q

backtest:
	$(PY) -m src.models.backtest

export:
	$(PY) -m src.export.powerbi

notebook:
	$(PY) scripts/build_eda_notebook.py

readme:
	$(PY) scripts/render_readme.py

clean:
	rm -rf data/processed/*.parquet reports/*.csv reports/*.json reports/figures/*.png powerbi/*.csv
