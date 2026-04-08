from anyio import Path
import polars as pl

SCHEMA = pl.Schema({
    "SimTime":            pl.UInt32,
    "StationId":          pl.UInt16,
    "TotalDeliveredKWh":  pl.Float32,
    "TotalMaxKWh":        pl.Float32,
    "TotalQueueSize":     pl.Int32,
    "Price":              pl.Float32,
    "TotalChargers":      pl.Int32,
    "Reservations":       pl.UInt32,
    "Cancellations":      pl.UInt32,
})

def load(path: str | Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    _validate_schema(df, path)
    return df


def _validate_schema(df: pl.DataFrame, path: str) -> None:
    actual = df.schema
    for col, expected_type in SCHEMA.items():
        if col not in actual:
            raise ValueError(f"[StationSnapshot] Missing column '{col}' in {path}")
        if actual[col] != expected_type:
            raise ValueError(
                f"[StationSnapshot] Column '{col}' has type {actual[col]}, "
                f"expected {expected_type} in {path}"
            )