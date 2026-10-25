"""Input helpers for user-provided tabular data."""
from __future__ import annotations

from io import BytesIO

import pandas as pd


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read a Streamlit upload, supporting common CSV encodings."""
    if uploaded_file is None:
        raise TypeError("uploaded_file must be a file-like object with a getvalue() method.")
    if not hasattr(uploaded_file, "getvalue"):
        raise TypeError("uploaded_file must be a file-like object with a getvalue() method.")

    content = uploaded_file.getvalue()
    if content is None or len(content) == 0:
        raise ValueError("The uploaded CSV is empty.")

    try:
        return pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(BytesIO(content), encoding="latin-1")
        except pd.errors.EmptyDataError as error:
            raise ValueError("The uploaded CSV must contain a header row.") from error
    except pd.errors.EmptyDataError as error:
        raise ValueError("The uploaded CSV must contain a header row.") from error