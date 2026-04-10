import polars as pl

CHARGER_SCHEMA: dict[str, pl.DataType] = {
    "SimTime":          pl.UInt32,
    "StationId":        pl.UInt16,
    "ChargerId":        pl.Int32,
    "MaxKWh":           pl.Float32,
    "QueueSize":        pl.Int32,
    "Utilization":      pl.Float32,
    "DeliveredKW":      pl.Float32,
    "IsDual":           pl.Boolean,
    "TargetEVDemandKW": pl.Float32,
}

STATION_SCHEMA: dict[str, pl.DataType] = {
    "SimTime":           pl.UInt32,
    "StationId":         pl.UInt16,
    "TotalDeliveredKWh": pl.Float32,
    "TotalMaxKWh":       pl.Float32,
    "TotalQueueSize":    pl.Int32,
    "Price":             pl.Float32,
    "TotalChargers":     pl.Int32,
    "Reservations":      pl.UInt32,
    "Cancellations":     pl.UInt32,
}

def validate_schema(df: pl.DataFrame, expected: dict[str, pl.DataType], label: str) -> None:
    errors: list[str] = []
    for col, expected_type in expected.items():
        if col not in df.schema:
            errors.append(f"  Missing column: '{col}'")
        elif df.schema[col] != expected_type:
            errors.append(f"  Wrong type for '{col}': expected {expected_type}, got {df.schema[col]}")

    if errors:
        raise ValueError(f"Schema validation failed for {label}:\n" + "\n".join(errors))