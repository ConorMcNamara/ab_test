.PHONY: install test lint format typecheck coverage check clean

# Install all dependencies including dev tools
install:
	uv sync --extra dev

# Run the full test suite
test:
	uv run pytest tests/ -v

# Run tests, skipping the slow simulation tests
test-fast:
	uv run pytest tests/ -v \
		--ignore=tests/frequentist_binomial/test_power_calculations.py \
		--ignore=tests/bayesian_binomial/test_bayesian_power_calculations.py \
		-k "not coverage"

# Check for lint errors (does not modify files)
lint:
	uv run ruff check ab_test/ tests/

# Auto-format and auto-fix lint issues
format:
	uv run ruff format ab_test/ tests/
	uv run ruff check --fix ab_test/ tests/

# Run tests and open an HTML coverage report
coverage:
	uv run pytest tests/ -v --cov-report=html
	open htmlcov/index.html

# Run zuban type checking
typecheck:
	uv run zuban check ab_test/

# Run lint + typecheck + tests (full quality gate)
check: lint typecheck test

# Remove Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
