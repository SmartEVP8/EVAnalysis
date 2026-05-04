"""
Defines the expected column names and data types for each raw simulation metric.

Every Parquet file loaded by the pipeline is checked against one of these
schemas before any analysis is performed.
"""

import polars as pl

CHARGER_SCHEMA: dict[str, pl.DataType] = {
    "SimTime":          pl.UInt32,
    "StationId":        pl.UInt16,
    "ChargerId":        pl.Int32,
    "MaxKWh":           pl.Float32,
    "Utilization":      pl.Float32,
}

STATION_SCHEMA: dict[str, pl.DataType] = {
    "SimTime":                     pl.UInt32,
    "StationId":                   pl.UInt16,
    "Price":                       pl.Float32,
    "TotalChargers":               pl.Int32,
    "Reservations":                pl.UInt32,
    "Cancellations":               pl.UInt32,
    "ExpectedWaitTimeMiliseconds": pl.UInt32,
}

ARRIVE_AT_DESTINATION_SCHEMA: dict[str, pl.DataType] = {
    "ExpectedArrivalTime":        pl.UInt32,
    "ActualArrivalTime":          pl.UInt32,
    "PathDeviation":              pl.Int32,
    "MissedDeadline":             pl.Boolean,
    "DriveDirectlyToDestination": pl.Boolean,
}


def validate_schema(
    df: pl.DataFrame,
    expected: dict[str, pl.DataType],
    label: str,
) -> None:
    """
    Validates that a DataFrame matches an expected schema.

    Checks that every required column is present and has the correct data type.
    Raises ValueError listing all mismatches at once so callers can fix them
    without repeated trial-and-error.
    """
    errors: list[str] = []

    for column, expected_type in expected.items():
        if column not in df.schema:
            errors.append(f"  Missing column: '{column}'")
        elif df.schema[column] != expected_type:
            errors.append(
                f"  Wrong type for '{column}': "
                f"expected {expected_type}, got {df.schema[column]}"
            )

    if errors:
        raise ValueError(
            f"Schema validation failed for '{label}':\n"
            + "\n".join(errors)
            + "\n\nCheck the simulation output or the loader configuration."
        )