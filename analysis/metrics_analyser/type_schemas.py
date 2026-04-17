"""
Defines the data types and columns for the analysis.
This module ensures that every Parquet file loaded into the system has the 
correct columns and data types.
"""

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

ARRIVE_AT_DESTINATION_SCHEMA: dict[str, pl.DataType] = {
    "ExpectedArrivalTime":      pl.UInt32,
    "ActualArrivalTime":        pl.UInt32,
    "PathDeviation":            pl.Int32,
    "DeltaArrivalTime":         pl.Int32,
    "MissedDeadline":           pl.Boolean,
}

def validate_schema(df: pl.DataFrame, expected: dict[str, pl.DataType], label: str) -> None:
    """
    Acts as a quality control check for incoming DataFrames.

    This function compares a loaded DataFrame against one of the defined
    schemas above. If a column is missing or a data type is wrong 
    (e.g., a number where a boolean should be), we raise an error.

    Args:
        df (pl.DataFrame): The data to check.
        expected (dict): The schema to compare against.
        label (str): A name for the data (used in the error message for clarity).

    Raises:
        ValueError: If the data does not match the expected schema.
    """
    errors: list[str] = []

    for col, expected_type in expected.items():
        if col not in df.schema:
            errors.append(f"  Missing column: '{col}'")
        elif df.schema[col] != expected_type:

            errors.append(f"  Wrong type for '{col}': expected {expected_type}, got {df.schema[col]}")

    if errors:
        raise ValueError(
            f"Schema validation failed for {label}:\n" + 
            "\n".join(errors) + 
            "\n\nPlease check the simulation output or the loader configuration."
        )