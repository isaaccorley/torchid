.PHONY: install check test clean

install:
	uv sync --extra cpu --group docs --group validation

check:
	uv run pre-commit run --all-files

test:
	uv run --extra cpu --group validation pytest --cov=src tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
