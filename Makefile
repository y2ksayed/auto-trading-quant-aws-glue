.PHONY: install install-dev test lint producer stream backtest dist publish-test

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check quantstream/ tests/ || true

producer:
	python -m quantstream.market_producer --rate 2 --duration 60

stream:
	S3_BUCKET=./data python -m quantstream.glue_streaming

backtest:
	python -m quantstream.quant --csv data/sample_ticks.csv

dist:
	python -m build

publish-test:
	twine check dist/* && twine upload --repository testpypi dist/*
