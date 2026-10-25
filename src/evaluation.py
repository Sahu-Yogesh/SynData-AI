"""Schema-aware fidelity evaluation independent of Streamlit rendering."""
from __future__ import annotations
import json
import time
from typing import Any
import numpy as np
import pandas as pd

def _wasserstein(a: pd.Series, b: pd.Series) -> float | None:
    a, b = pd.to_numeric(a, errors="coerce").dropna().to_numpy(), pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if not len(a) or not len(b): return None
    a, b = np.sort(a), np.sort(b); points = np.sort(np.concatenate([a, b]))
    if points[-1] == points[0]: return 0.0
    return float(np.sum(np.abs(np.searchsorted(a, points[:-1], side="right") / len(a) - np.searchsorted(b, points[:-1], side="right") / len(b)) * np.diff(points)))

def evaluate_datasets(real: pd.DataFrame, synthetic: pd.DataFrame, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    if not isinstance(real, pd.DataFrame) or not isinstance(synthetic, pd.DataFrame): raise TypeError("real and synthetic must be DataFrames")
    real_cols, synth_cols, common = list(real.columns), list(synthetic.columns), [c for c in real.columns if c in synthetic.columns]
    schema = {"columns_match": real_cols == synth_cols, "missing_in_synthetic": [c for c in real_cols if c not in synthetic.columns], "unexpected_in_synthetic": [c for c in synth_cols if c not in real.columns], "common_columns": common, "dtype_compatibility": {str(c): str(real[c].dtype) == str(synthetic[c].dtype) for c in common}, "row_counts": {"real": len(real), "synthetic": len(synthetic)}}
    numeric, categorical = [], []
    for c in common:
        if pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(synthetic[c]):
            r, s = pd.to_numeric(real[c], errors="coerce"), pd.to_numeric(synthetic[c], errors="coerce"); rq, sq = r.quantile([.25, .5, .75]), s.quantile([.25, .5, .75]); wd = _wasserstein(r, s); scale = max(float(r.max() - r.min()) if r.notna().any() else 0., 1e-12)
            numeric.append({"column": str(c), "real_mean": float(r.mean()), "synthetic_mean": float(s.mean()), "mean_difference": float(abs(r.mean() - s.mean())), "real_std": float(r.std()) if r.notna().sum() > 1 else 0., "synthetic_std": float(s.std()) if s.notna().sum() > 1 else 0., "median_difference": float(abs(rq[.5] - sq[.5])), "quantile_difference": float(np.mean(abs(rq - sq))), "wasserstein_distance": wd, "normalized_wasserstein": wd / scale if wd is not None else None, "real_missing": int(r.isna().sum()), "synthetic_missing": int(s.isna().sum())})
        else:
            rf = real[c].astype("object").where(real[c].notna(), "<MISSING>").value_counts(normalize=True); sf = synthetic[c].astype("object").where(synthetic[c].notna(), "<MISSING>").value_counts(normalize=True); keys = set(rf.index) | set(sf.index); tvd = .5 * sum(abs(float(rf.get(k, 0)) - float(sf.get(k, 0))) for k in keys)
            categorical.append({"column": str(c), "real_unique": int(real[c].nunique(dropna=False)), "synthetic_unique": int(synthetic[c].nunique(dropna=False)), "total_variation_distance": float(tvd), "distribution_similarity": float(1 - tvd), "missing_difference": int(real[c].isna().sum() - synthetic[c].isna().sum()), "categories_only_in_synthetic": [str(k) for k in set(sf.index) - set(rf.index)]})
    num_cols = [x["column"] for x in numeric]; corr = {"columns": num_cols, "real": {}, "synthetic": {}, "absolute_difference": {}, "mean_absolute_difference": None}
    if len(num_cols) >= 2:
        rc, sc = real[num_cols].corr(), synthetic[num_cols].corr(); diff = (rc - sc).abs().fillna(0.); np.fill_diagonal(diff.values, 0); denom = len(num_cols) * (len(num_cols) - 1); corr.update({"real": rc.to_dict(), "synthetic": sc.to_dict(), "absolute_difference": diff.to_dict(), "mean_absolute_difference": float(diff.to_numpy().sum() / denom)})
    report = {"metadata": metadata or {}, "schema": schema, "numerical_fidelity": numeric, "categorical_fidelity": categorical, "correlation": corr, "sdv_quality": None, "warnings": [], "duration_seconds": time.perf_counter() - start, "errors": []}
    if not schema["columns_match"]: report["warnings"].append("Schema mismatch: statistical comparisons use common columns only.")
    return report

def add_sdv_quality(report: dict[str, Any], real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    try:
        from sdv.metadata import Metadata
        from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
        metadata = Metadata.detect_from_dataframe(data=real, table_name="evaluation_table"); diagnostic = run_diagnostic(real_data=real, synthetic_data=synthetic, metadata=metadata); quality = evaluate_quality(real_data=real, synthetic_data=synthetic, metadata=metadata)
        report["sdv_quality"] = {"overall_score": float(quality.get_score()), "properties": quality.get_properties().to_dict(orient="records"), "diagnostic_properties": diagnostic.get_properties().to_dict(orient="records")}
    except Exception as error: report["warnings"].append(f"SDV evaluation unavailable or failed: {error}")
    return report

def report_json(report: dict[str, Any]) -> str: return json.dumps(report, indent=2, default=str)
