"""
Silver layer schema — one row per NUTS 2 region, all features harmonized.
Validated after Phase 2 completes. Primary key: nuts2_code (index, unique).
"""

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Index

silver_schema = DataFrameSchema(
    index=Index(
        pa.String,
        name="nuts2_code",
        checks=[
            pa.Check(lambda s: s.str.match(r"^[A-Z]{2}[A-Z0-9]{2}$").all(),
                     error="nuts2_code must match NUTS 2 format, e.g. DE21"),
        ],
        unique=True,
        nullable=False,
    ),
    columns={
        # --- Identifiers ---
        "nuts2_name": Column(pa.String, nullable=False),
        "country_code": Column(
            pa.String,
            nullable=False,
            checks=pa.Check.str_length(2, 2),
        ),

        # --- Cost dimension ---
        "gdp_per_capita_pps": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.greater_than(0),
        ),
        "avg_wage_eur_ppp": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.greater_than(0),
        ),

        # --- Talent dimension ---
        "hrst_per_1000": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 1000),
        ),
        "tertiary_enrolment_rate": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 100),
        ),

        # --- Infrastructure dimension ---
        "broadband_penetration_pct": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 100),
        ),

        # --- Cluster effects dimension ---
        "rd_expenditure_pct_gdp": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 15),
        ),
        "business_rd_pct_gdp": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 10),
        ),
        "epo_patents_per_mio_pop": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.greater_than_or_equal_to(0),
        ),
        "horizon_eu_projects_total": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.greater_than_or_equal_to(0),
        ),
        "horizon_eu_funding_eur": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.greater_than_or_equal_to(0),
        ),
        "coordinator_dummy": Column(
            pa.Int,
            nullable=True,
            checks=pa.Check.isin([0, 1]),
        ),
        "startup_density_proxy": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.greater_than_or_equal_to(0),
        ),
        "renewable_energy_share": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 100),
        ),

        # --- Location Quotients (capped at 5.0) ---
        "lq_nace_j62j63": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 5.001),
        ),
        "lq_nace_c21_m72": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 5.001),
        ),
        "lq_nace_c26": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 5.001),
        ),
        "lq_nace_d35_clean": Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0, 5.001),
        ),

        # --- Population ---
        "population": Column(
            pa.Float,
            nullable=False,
            checks=pa.Check.greater_than(0),
        ),

        # --- Data quality ---
        "data_quality_score": Column(
            pa.Float,
            nullable=False,
            checks=pa.Check.in_range(0, 1),
        ),

        # --- Reference years (one per variable group) ---
        "ref_year_cost": Column(pa.Int, nullable=True),
        "ref_year_talent": Column(pa.Int, nullable=True),
        "ref_year_infra": Column(pa.Int, nullable=True),
        "ref_year_cluster": Column(pa.Int, nullable=True),
    },
    strict=False,  # imputed_flag and mnar_flag columns are dynamic; allow extras
    coerce=True,
    name="silver_layer",
)
