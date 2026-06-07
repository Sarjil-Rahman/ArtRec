from __future__ import annotations
from pathlib import Path
import ast
import pandas as pd

LIST_COLUMNS = {"embedding", "taste_vector"}
DATETIME_COLUMNS = {"timestamp"}


def _parse_possible_list(value):
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def read_csv_with_types(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in set(df.columns).intersection(LIST_COLUMNS):
        df[col] = df[col].apply(_parse_possible_list)
    for col in set(df.columns).intersection(DATETIME_COLUMNS):
        df[col] = pd.to_datetime(df[col])
    return df


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
