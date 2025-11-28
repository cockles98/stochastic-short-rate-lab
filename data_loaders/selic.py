"""Utilities to load daily SELIC data from CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_selic_csv(path: Path | str) -> pd.DataFrame:
    """Load the Bacen SELIC CSV into a clean DataFrame."""

    file_path = Path(path)
    raw = file_path.read_text(encoding="latin-1").splitlines()
    # Skip initial descriptive lines until header row starting with Data;
    data_lines = [line for line in raw if line.startswith("Data;")]
    if not data_lines:
        raise ValueError("Cabeçalho 'Data;' não encontrado no arquivo SELIC.")
    start_idx = raw.index(data_lines[0])
    csv_text = "\n".join(raw[start_idx:])
    df = pd.read_csv(pd.io.common.StringIO(csv_text), sep=";")
    df = df.rename(
        columns={
            "Data": "date",
            "Taxa (% a.a.)": "rate_annual",
            "Fator diário": "daily_factor",
        }
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["rate_annual"] = (
        df["rate_annual"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
        / 100.0
    )
    if "daily_factor" in df.columns:
        df["daily_factor"] = (
            df["daily_factor"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
    else:
        df["daily_factor"] = 1.0 + df["rate_annual"] / 252.0
    return df.dropna(subset=["date"])


def get_latest_rate(df: pd.DataFrame) -> float:
    """Return the most recent SELIC rate in fraction."""

    latest = df.sort_values("date").iloc[-1]
    return float(latest["rate_annual"])


def get_window(df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Filter the SELIC series by date window."""

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end) if end else df["date"].max()
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    return df.loc[mask].copy()
