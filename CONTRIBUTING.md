# Contributing

Contributions are welcome. Please follow the guidelines below to keep the codebase consistent.

## Development Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/ConorMcNamara/ab_test.git
   cd ab_test
   ```

2. **Install dependencies** (including dev tools)

   ```bash
   poetry install --extras dev
   ```

3. **Verify the setup**

   ```bash
   make test
   ```

## Workflow

1. Create a branch off `main`:

   ```bash
   git checkout -b your-feature-name
   ```

2. Make your changes, following the [code standards](#code-standards) below.

3. Add or update tests in `tests/` to cover your changes.

4. Run the full quality check before pushing:

   ```bash
   make check
   ```

5. Open a pull request against `main`. Fill in a brief description of what changed and why.

## Code Standards

### Style

This project uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting. The configuration lives in `pyproject.toml`.

```bash
make format   # auto-format
make lint     # check for lint errors
```

The key rules enforced are:

- `E` / `F` — standard pycodestyle and pyflakes errors
- `UP` — pyupgrade: modern Python syntax (e.g., `X | Y` instead of `Union[X, Y]`)

### Type annotations

- All public functions must have fully annotated signatures.
- Use Python 3.10+ union syntax (`X | Y`, `X | None`) — do **not** import `Union` or `Optional` from `typing`.
- Use built-in generics (`list[str]`, `dict[str, int]`) — do **not** import `List`, `Dict`, etc. from `typing`.
- Use `np.ndarray` for NumPy array type hints, not `np.array`.

### Tests

- Tests live in `tests/binomial/` and mirror the source layout.
- Use `pytest`. Run with `make test`.
- Every new public function or method needs at least one test.
- Avoid adding new slow tests (marked `# @pytest.mark.slow`) without good reason.

## Project Layout

```
ab_test/
├── ab_test/
│   └── binomial/
│       ├── confidence_intervals.py   # CI methods (Wilson, Agresti-Coull, etc.)
│       ├── contingency.py            # ContingencyTable class
│       ├── power_calculations.py     # Power, MDL, required sample size
│       ├── stats_tests.py            # Significance tests
│       └── utils.py                  # MLE, observed lift, Wilson significance
└── tests/
    └── binomial/
        ├── test_confidence_intervals.py
        ├── test_contingency.py
        ├── test_power_calculations.py
        ├── test_stats_test.py
        └── test_utils.py
```

## Reporting Bugs

Open an issue on GitHub with:

- A minimal reproducible example
- The Python and library version (`python --version`, `poetry show ab-test`)
- The full traceback
