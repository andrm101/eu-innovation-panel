"""
Gold layer schema — Silver plus PCA composite scores, archetype assignments,
and clustering metadata. Validated after Phase 5 completes.
"""

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Index

gold_schema = DataFrameSchema(
    index=Index(
        pa.String,
        name="nuts2_code",
        checks=[
            pa.Check(lambda s: s.str.match(r"^[A-Z]{2}[A-Z0-9]{2}$").all(),
                     error="nuts2_code must match NUTS 2 format"),
        ],
        unique=True,
        nullable=False,
    ),
    columns={
        # --- Composite dimension scores (z-scored, for clustering) ---
        "score_cost": Column(pa.Float, nullable=True),
        "score_talent": Column(pa.Float, nullable=True),
        "score_infra": Column(pa.Float, nullable=True),
        "score_cluster": Column(pa.Float, nullable=True),

        # --- MinMax-normalized scores (for radar visualization only) ---
        "score_cost_minmax": Column(
            pa.Float, nullable=True, checks=pa.Check.in_range(-0.001, 1.001)
        ),
        "score_talent_minmax": Column(
            pa.Float, nullable=True, checks=pa.Check.in_range(-0.001, 1.001)
        ),
        "score_infra_minmax": Column(
            pa.Float, nullable=True, checks=pa.Check.in_range(-0.001, 1.001)
        ),
        "score_cluster_minmax": Column(
            pa.Float, nullable=True, checks=pa.Check.in_range(-0.001, 1.001)
        ),

        # --- LQ values (capped at 5.0) ---
        "lq_nace_j62j63": Column(
            pa.Float, nullable=True, checks=pa.Check.in_range(0, 5.001)
        ),
        "lq_nace_c21_m72": Column(
            pa.Float, nullable=True, checks=pa.Check.in_range(0, 5.001)
        ),
        "lq_nace_c26": Column(
            pa.Float, nullable=True, checks=pa.Check.in_range(0, 5.001)
        ),
        "lq_nace_d35_clean": Column(
            pa.Float, nullable=True, checks=pa.Check.in_range(0, 5.001)
        ),

        # --- Clustering outputs ---
        "archetype_id": Column(pa.Int, nullable=True),
        "archetype_label": Column(pa.String, nullable=True),
        "distance_to_centroid": Column(
            pa.Float, nullable=True, checks=pa.Check.greater_than_or_equal_to(0)
        ),
        "silhouette_sample": Column(
            pa.Float, nullable=True, checks=pa.Check.in_range(-1.001, 1.001)
        ),
        "is_transition_region": Column(pa.Bool, nullable=False),
        "is_insufficient_data": Column(pa.Bool, nullable=False),

        # --- Data quality ---
        "data_quality_score": Column(
            pa.Float, nullable=False, checks=pa.Check.in_range(0, 1)
        ),
        "dimensionality_warning": Column(pa.Bool, nullable=False),

        # --- Pass-through identifiers ---
        "nuts2_name": Column(pa.String, nullable=False),
        "country_code": Column(pa.String, nullable=False, checks=pa.Check.str_length(2, 2)),
        "population": Column(pa.Float, nullable=False, checks=pa.Check.greater_than(0)),
    },
    strict=False,  # Silver columns pass through
    coerce=True,
    name="gold_layer",
)
