.PHONY: format lint test

format:
	python -m ruff format .

lint:
	python -m ruff check .

test:
	python -m pytest

