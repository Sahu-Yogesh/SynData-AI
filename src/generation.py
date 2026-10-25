"""Unified generation contract for supported reference-data methods."""
from __future__ import annotations
import time
from typing import Any
import numpy as np
import pandas as pd
from .ctgan_generator import train_ctgan_model, generate_ctgan_data

def _validate(data: pd.DataFrame, rows: int) -> None:
    if not isinstance(data, pd.DataFrame) or data.empty: raise ValueError("Reference data must be a non-empty DataFrame")
    if rows < 1: raise ValueError("Requested rows must be greater than zero")
    if data.columns.empty or data.columns.duplicated().any(): raise ValueError("Reference data must have unique column names")
    unsupported = [str(c) for c in data.columns if pd.api.types.is_complex_dtype(data[c])]
    if unsupported: raise ValueError(f"Unsupported complex columns: {unsupported}")

def _empirical_generate(data: pd.DataFrame, rows: int, seed: int | None) -> pd.DataFrame:
    rng, result = np.random.default_rng(seed), {}
    for name in data.columns:
        series, observed = data[name], data[name].dropna()
        if observed.empty: result[name] = pd.Series([pd.NA] * rows, dtype=series.dtype); continue
        values = rng.choice(observed.to_numpy(), size=rows, replace=True)
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 1:
            scale = float(series.std()) if pd.notna(series.std()) else 0.0; values = values + rng.normal(0, scale * .02, rows)
            if pd.api.types.is_integer_dtype(series): values = np.rint(values).astype(series.dtype)
        result[name] = values
    return pd.DataFrame(result, columns=data.columns).astype({c: data[c].dtype for c in data.columns}, errors="ignore")

def generate_synthetic(data: pd.DataFrame, method: str, rows: int, *, seed: int | None = None, model_path: str | None = None, epochs: int = 300) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate records and return a consistent ``(data, metadata)`` pair."""
    _validate(data, int(rows)); start = time.perf_counter(); key = method.strip().lower().replace(" ", "_")
    info: dict[str, Any] = {"method": method, "requested_rows": int(rows), "seed": seed, "error": None}
    try:
        if seed is not None: np.random.seed(seed)
        if key in {"custom", "custom_generator"}: synthetic = _empirical_generate(data, int(rows), seed)
        elif key in {"gaussian_copula", "gaussian_copula_synthesizer"}:
            from sdv.metadata import Metadata
            from sdv.single_table import GaussianCopulaSynthesizer
            model = GaussianCopulaSynthesizer(Metadata.detect_from_dataframe(data=data.copy(), table_name="syn_data")); model.fit(data.copy()); synthetic = model.sample(num_rows=int(rows))
        elif key in {"ctgan", "ctgan_generator"}:
            if not model_path: raise ValueError("model_path is required when training CTGAN")
            model, _ = train_ctgan_model(data.copy(), model_path, epochs=int(epochs)); synthetic = generate_ctgan_data(model, int(rows))
        else: raise ValueError(f"Unsupported generation method: {method}")
        synthetic = synthetic.reindex(columns=data.columns); info.update({"actual_rows": len(synthetic), "duration_seconds": time.perf_counter() - start}); return synthetic, info
    except Exception as error:
        info.update({"actual_rows": 0, "duration_seconds": time.perf_counter() - start, "error": str(error)}); raise
