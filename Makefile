.PHONY: install test lint format typecheck check clean

# Install all dependencies including dev tools
install:
	poetry install --extras dev

# Run the full test suite
test:
	poetry run pytest tests/ -v

# Run tests, skipping the slow simulation tests
test-fast:
	poetry run pytest tests/ -v \
		--ignore=tests/binomial/test_power_calculations.py \
		-k "not coverage"

# Check for lint errors (does not modify files)
lint:
	poetry run ruff check ab_test/ tests/

# Auto-format and auto-fix lint issues
format:
	poetry run ruff format ab_test/ tests/
	poetry run ruff check --fix ab_test/ tests/

# Run mypy type checking
typecheck:
	poetry run mypy ab_test/

# Run lint + typecheck + tests (full quality gate)
check: lint typecheck test

# Remove Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
