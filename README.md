# SynData AI

SynData AI is a Streamlit application for profiling tabular CSV data, generating synthetic records, and evaluating statistical fidelity.

## Setup and run

1. Create/repair a Python environment (the checked-in `venv` may need to be recreated if its original Python installation is unavailable).
2. Install the pinned dependencies:

   `pip install -r requirements.txt`

3. Start the application:

   `streamlit run app.py`

The default dashboard continues to use `data/telco_cleaned.csv`. The **Profiler & Experiment** page accepts any valid tabular CSV.

## Features

- Intelligent profiling reports dimensions, memory, dtypes, missingness, duplicates, constants, near-constants, IQR outliers, rare categories, identifier-like cardinality, categorical distributions, numerical summaries, and correlations. Facts and heuristic warnings are separated; no rows or columns are automatically removed.
- A unified generation workflow supports the empirical Custom Generator, SDV Gaussian Copula, and the existing SDV CTGAN implementation. Generation metadata records method, seed, requested/actual rows, duration, and errors. CTGAN models are saved with the existing model library conventions.
- Unified evaluation checks schema compatibility and compares numeric means, standard deviations, medians, quantiles, Wasserstein distance, categorical total variation distance, missingness, numerical correlations, and optional SDV diagnostic/quality reports. Reports can be downloaded as JSON and generated data as CSV.

## Metrics and limitations

Wasserstein and total-variation distances are differences, so lower is better; distribution similarity is `1 - TVD`. Correlation error is mean absolute off-diagonal correlation difference. SDV quality is the score returned by the installed SDV version and is not a privacy guarantee or a guarantee of downstream ML utility. Privacy screening remains a separate exact-match heuristic.

The Custom Generator is an empirical baseline for quick deterministic experiments, not a multivariate generative model. Gaussian Copula and CTGAN require compatible SDV metadata and may fail on unsupported or very small schemas. Review profiler warnings and decide explicitly how identifiers, missing values, rare categories, and outliers should be handled.

## Tests

Run the dependency-backed tests with:

`python -m unittest discover -s tests -v`

The current repository environment points to a removed Python 3.13 executable, so recreate it before running the app or dependency-backed tests.
