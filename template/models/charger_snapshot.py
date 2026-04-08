from pathlib import Path

import polars as pl

SCHEMA = pl.Schema({
    "SimTime":          pl.UInt32,
    "StationId":        pl.UInt16,
    "ChargerId":        pl.Int32,
    "MaxKWh":           pl.Float32,
    "QueueSize":        pl.Int32,
    "Utilization":      pl.Float32,
    "DeliveredKW":      pl.Float32,
    "IsDual":           pl.Boolean,
    "TargetEVDemandKW": pl.Float32,
})


def load(path: str | Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    _validate_schema(df, path)
    return df


def _validate_schema(df: pl.DataFrame, path: str) -> None:
    actual = df.schema
    for col, expected_type in SCHEMA.items():
        if col not in actual:
            raise ValueError(f"[ChargerSnapshot] Missing column '{col}' in {path}")
        if actual[col] != expected_type:
            raise ValueError(
                f"[ChargerSnapshot] Column '{col}' has type {actual[col]}, "
                f"expected {expected_type} in {path}"
            )