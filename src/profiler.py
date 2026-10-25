"""Reusable, defensive profiling for arbitrary tabular data."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
        non_missing = series.dropna()
        if not non_missing.empty and all(isinstance(value, (bool, np.bool_)) for value in non_missing):
            return "boolean"
        return "categorical"
    return "other"

def profile_dataset(data: pd.DataFrame, *, rare_threshold: float = .01, near_constant_threshold: float = .01, high_cardinality_threshold: float = .95) -> dict[str, Any]:
    """Return structured facts and heuristic warnings without modifying data."""
    if not isinstance(data, pd.DataFrame): raise TypeError("data must be a pandas DataFrame")
    if not 0 <= rare_threshold <= 1 or not 0 <= near_constant_threshold <= 1 or not 0 <= high_cardinality_threshold <= 1:
        raise ValueError("proportion thresholds must be between 0 and 1")
    rows, columns = data.shape
    column_rows, numerical, categorical, outliers, rare_values, warnings = [], [], [], [], [], []
    for name in data.columns:
        series, kind = data[name], _column_kind(data[name])
        missing, unique = int(series.isna().sum()), int(series.nunique(dropna=False))
        frequencies = series.astype("object").where(series.notna(), "<MISSING>").value_counts(dropna=False)
        top_frequency = int(frequencies.iloc[0]) if len(frequencies) else 0
        info = {"column": str(name), "dtype": str(series.dtype), "kind": kind, "missing_count": missing, "missing_percent": missing / rows * 100 if rows else 0.0, "unique_count": unique, "unique_percent": unique / rows if rows else 0.0, "top_value": str(frequencies.index[0]) if len(frequencies) else None, "top_frequency": top_frequency, "is_constant": unique <= 1 and rows > 0, "is_near_constant": bool(rows and top_frequency / rows >= 1 - near_constant_threshold), "is_identifier_like": bool(rows and unique / rows >= high_cardinality_threshold and unique > 10)}
        column_rows.append(info)
        if kind == "numerical":
            values = pd.to_numeric(series, errors="coerce").dropna()
            if len(values):
                q1, median, q3 = values.quantile([.25, .5, .75]); iqr = float(q3 - q1); low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
                count = int(((values < low) | (values > high)).sum()) if iqr > 0 else 0
                outliers.append({"column": str(name), "count": count, "percent": count / len(values) * 100, "lower_bound": low, "upper_bound": high})
                numerical.append({"column": str(name), "count": int(values.size), "mean": float(values.mean()), "median": float(median), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "min": float(values.min()), "q1": float(q1), "q3": float(q3), "max": float(values.max()), "skew": float(values.skew()) if len(values) > 2 else None})
        elif kind in {"categorical", "boolean"}:
            total = max(1, int(frequencies.sum()))
            values = [{"value": str(k), "count": int(v), "percent": float(v / total * 100)} for k, v in frequencies.items()]
            categorical.append({"column": str(name), "unique_count": unique, "values": values[:100], "imbalance_ratio": float(frequencies.iloc[0] / total) if len(frequencies) else 0.0})
            for value, count in frequencies.items():
                if rows and count / rows < rare_threshold: rare_values.append({"column": str(name), "value": str(value), "count": int(count), "percent": float(count / rows * 100)})
        if info["is_constant"]: warnings.append({"type": "fact", "column": str(name), "message": "Column is constant."})
        if info["is_identifier_like"]: warnings.append({"type": "heuristic", "column": str(name), "message": "High cardinality may indicate an identifier-like column."})
        if missing: warnings.append({"type": "fact", "column": str(name), "message": "Missing values require explicit handling."})
    if int(data.duplicated().sum()): warnings.append({"type": "fact", "column": "<dataset>", "message": "Duplicate rows were detected."})
    if rows == 0: warnings.append({"type": "fact", "column": "<dataset>", "message": "Dataset has no rows; generation and evaluation cannot proceed."})
    numeric_names = [name for name in data.columns if _column_kind(data[name]) == "numerical"]
    correlation = data[numeric_names].corr().replace({np.nan: None}).to_dict() if len(numeric_names) >= 2 else {}
    return {"overview": {"rows": rows, "columns": columns, "memory_bytes": int(data.memory_usage(deep=True).sum()), "duplicate_rows": int(data.duplicated().sum()), "empty": bool(data.empty)}, "columns": column_rows, "numerical": numerical, "categorical": categorical, "outliers": outliers, "rare_values": rare_values, "correlation": correlation, "warnings": warnings, "recommendations": ["Review missing values before training.", "Review identifier-like columns before generation.", "Inspect outliers and rare categories before training."]}


