"""Baixa série da curva DI/OTN a partir de API pública do Bacen."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4390/dados"  # DI - taxa swap pré


def fetch_data(start: str | None = None, end: str | None = None) -> list[dict[str, str]]:
    params = {"formato": "json"}
    if start:
        params["dataInicial"] = start
    if end:
        params["dataFinal"] = end
    response = requests.get(BCB_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_records(records: Iterable[dict[str, str]]) -> pd.DataFrame:
    rows = []
    for item in records:
        try:
            date = datetime.strptime(item["data"], "%d/%m/%Y").date()
            value = float(item["valor"].replace(",", ".")) / 100.0  # converter para fração
        except (KeyError, ValueError):
            continue
        rows.append({"date": date, "rate": value})
    if not rows:
        raise ValueError("Nenhum registro válido encontrado.")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Data inicial (dd/mm/aaaa)")
    parser.add_argument("--end", help="Data final (dd/mm/aaaa)")
    parser.add_argument(
        "--out", default="data/raw_di_curve.parquet", help="Caminho de saída em parquet/csv"
    )
    parser.add_argument(
        "--force-csv", action="store_true", help="Salvar sempre em CSV (ignora sufixo)"
    )
    args = parser.parse_args()

    records = fetch_data(start=args.start, end=args.end)
    df = normalize_records(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.force_csv or out_path.suffix == ".csv":
        out_path = out_path.with_suffix(".csv") if args.force_csv else out_path
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    print(f"Dados salvos em {out_path} ({len(df)} linhas)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entry
        print(f"Erro ao baixar/normalizar dados: {exc}", file=sys.stderr)
        sys.exit(1)
