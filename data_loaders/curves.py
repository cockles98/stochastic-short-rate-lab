"""Parsers for real-world zero-coupon curve CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class CurveComponents:
    ref_date: str
    params: pd.DataFrame
    ettj: pd.DataFrame
    prefixados: pd.DataFrame
    residuals: pd.DataFrame


def _split_sections(lines: list[str]) -> list[list[str]]:
    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                sections.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        sections.append(current)
    return sections


def _parse_params(section: list[str]) -> tuple[str, pd.DataFrame]:
    header = section[0]
    ref_date = header.split(";")[0].strip()
    data_rows = section[1:]
    records = []
    for row in data_rows:
        fields = row.split(";")
        curve_type = fields[0]
        values = [fields[0]] + [field.replace(",", ".") for field in fields[1:]]
        records.append(values)
    columns = ["curve_type", "beta1", "beta2", "beta3", "beta4", "lambda1", "lambda2"]
    df = pd.DataFrame(records, columns=columns)
    numeric_cols = columns[1:]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return ref_date, df


def _extract_table_lines(section: list[str], header_prefix: str) -> list[str]:
    for idx, line in enumerate(section):
        if line.lower().startswith(header_prefix.lower()):
            return section[idx:]
    raise ValueError(f"Header '{header_prefix}' não encontrado na seção.")


def _parse_table(lines: list[str], columns: list[str]) -> pd.DataFrame:
    header = lines[0].split(";")
    data = [row.split(";") for row in lines[1:]]
    df = pd.DataFrame(data, columns=header)
    df = df.replace("", np.nan).dropna(how="all")
    df.columns = columns
    for col in columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_residuals(section: list[str]) -> pd.DataFrame:
    header = section[1].split(";")
    data = [row.split(";") for row in section[2:]]
    df = pd.DataFrame(data, columns=header)
    df["SELIC"] = df["SELIC"].str.replace(".", "", regex=False).astype(float, errors="ignore")
    df["Erro (%a.a.)"] = (
        df["Erro (%a.a.)"]
        .str.replace(",", ".", regex=False)
        .astype(float, errors="ignore")
    )
    return df


def load_curve_components(path: Path | str) -> CurveComponents:
    """Parse CurvaZero_*.csv into structured components."""

    file_path = Path(path)
    try:
        raw_text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw_text = file_path.read_text(encoding="latin-1")
    content = raw_text.splitlines()
    sections = _split_sections(content)
    if len(sections) < 4:
        raise ValueError("Arquivo de curva incompleto: seções insuficientes.")

    ref_date, params_df = _parse_params(sections[0])

    ettj_lines = _extract_table_lines(sections[1], "Vertices")
    ettj_columns = ["Vertices", "ETTJ IPCA", "ETTJ PREF", "Inflacao Implicita"]
    ettj_df = _parse_table(ettj_lines, ettj_columns)

    prefix_lines = _extract_table_lines(sections[2], "Vertices")
    prefix_columns = ["Vertices", "Taxa (%a.a.)"]
    prefix_df = _parse_table(prefix_lines, prefix_columns)

    residuals_df = _parse_residuals(sections[3])

    return CurveComponents(
        ref_date=ref_date,
        params=params_df,
        ettj=ettj_df,
        prefixados=prefix_df,
        residuals=residuals_df,
    )
