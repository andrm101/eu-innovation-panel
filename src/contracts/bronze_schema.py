"""
Bronze layer schema — minimal validation applied to raw ingested files.
Bronze data is immutable; we only assert structural integrity, not value ranges.
"""

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Index

# Minimal schema applied to every raw Eurostat TSV after initial parse.
# Each source gets its own parse step; this validates the common fields
# present in every Eurostat bulk download after melting to long format.

eurostat_long_bronze = DataFrameSchema(
    columns={
        "geo": Column(
            pa.String,
            nullable=False,
            description="NUTS code as provided in source (may be 2016 or 2021 vintage)",
        ),
        "time": Column(
            pa.Int,
            nullable=False,
            checks=pa.Check.in_range(2000, 2030),
            description="Reference year",
        ),
        "value": Column(
            pa.Float,
            nullable=True,
            description="Observed value; NaN is permitted at Bronze layer",
        ),
        "unit": Column(
            pa.String,
            nullable=True,
            description="Unit of measure as provided by source",
        ),
        "dataset_code": Column(
            pa.String,
            nullable=False,
            description="Eurostat dataset code, e.g. rd_e_gerdreg",
        ),
        "source_file": Column(
            pa.String,
            nullable=False,
            description="Relative path to the raw file in data/bronze/",
        ),
    },
    strict=False,  # allow extra columns from source
    coerce=True,
    name="eurostat_long_bronze",
)

# Generic manifest schema applied to bronze_manifest.json entries when loaded as a DataFrame
bronze_manifest_schema = DataFrameSchema(
    columns={
        "file_path": Column(pa.String, nullable=False),
        "sha256": Column(pa.String, nullable=False, checks=pa.Check.str_length(64, 64)),
        "download_timestamp": Column(pa.String, nullable=False),
        "source_url": Column(pa.String, nullable=True),
        "dataset_code": Column(pa.String, nullable=True),
        "size_bytes": Column(pa.Int, nullable=False, checks=pa.Check.greater_than(0)),
    },
    strict=False,
    coerce=True,
    name="bronze_manifest",
)
