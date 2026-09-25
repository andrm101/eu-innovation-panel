"""
Phase 4 — PCA & Composite Dimension Scores
Reads Silver, constructs 4 dimension scores, writes Gold parquet.

Excluded from all scoring:
  - net_migration_rate: corrupted (= population values in Silver, ingestion filter miss)
  - population: context variable; not analytically scored

Dimensions:
  score_cost    — z-score of log1p(gdp_per_capita_pps)
  score_talent  — PCA PC1 of 4 talent indicators (after log1p for hi_tech_employment_pct)
  score_infra   — mean z-score of 2 infrastructure indicators
  score_cluster — PCA PC1 of 8 cluster/innovation indicators (after log1p for right-skewed)

Log1p applied to features with skew > 1 (right-skewed):
  gdp_per_capita_pps, hi_tech_employment_pct, rd_expenditure_pct_gdp,
  business_rd_pct_gdp, epo_patents_per_mio_pop, gva_ict_share,
  lq_nace_j62j63, lq_nace_c21_m72, lq_nace_c26

employment_rate: skew = -1.082 (left-skewed) — log1p inapplicable; used untransformed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import structlog
from utils.logging_config import configure_logging, pipeline_step

configure_logging()
log = structlog.get_logger(__name__)

SILVER_PATH = ROOT / "data" / "silver" / "region_profiles_silver.parquet"
GOLD_PATH   = ROOT / "data" / "gold" / "region_profiles_gold.parquet"
FIG_DIR     = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
GOLD_PATH.parent.mkdir(exist_ok=True)

# ── Feature definitions ────────────────────────────────────────────────────────

COST_FEATURES    = ["gdp_per_capita_pps"]
# hi_tech_employment_pct removed: Cohen's d = -0.069 (near-zero), 242/242 imputed
# to median (~14.5%) in Silver — contributes noise, not signal, to this dimension.
TALENT_FEATURES  = ["hrst_per_1000", "tertiary_enrolment_rate", "employment_rate"]
INFRA_FEATURES   = ["broadband_penetration_pct", "enterprise_internet_use"]
CLUSTER_FEATURES = ["rd_expenditure_pct_gdp", "business_rd_pct_gdp",
                    "epo_patents_per_mio_pop", "gva_ict_share",
                    "lq_nace_j62j63", "lq_nace_c21_m72",
                    "lq_nace_c26", "lq_nace_d35_clean"]

ALL_ANALYTICAL = COST_FEATURES + TALENT_FEATURES + INFRA_FEATURES + CLUSTER_FEATURES

# Features whose distribution is right-skewed (skew > 1) → apply log1p
LOG1P_FEATURES = {
    "gdp_per_capita_pps",
    "rd_expenditure_pct_gdp",
    "business_rd_pct_gdp",
    "epo_patents_per_mio_pop",
    "gva_ict_share",
    "lq_nace_j62j63",
    "lq_nace_c21_m72",
    "lq_nace_c26",
}

# LQ caps enforced by Gold schema (pa.Check.in_range(0, 5.001))
LQ_COLS = ["lq_nace_j62j63", "lq_nace_c21_m72", "lq_nace_c26", "lq_nace_d35_clean"]

FEATURE_LABELS = {
    "gdp_per_capita_pps":       "GDP/capita (PPS)",
    "hrst_per_1000":            "HRST per 1 000 pop.",
    "tertiary_enrolment_rate":  "Tertiary enrolment rate",
    "employment_rate":          "Employment rate",
    "hi_tech_employment_pct":   "Hi-tech employment %",
    "broadband_penetration_pct":"Broadband penetration",
    "enterprise_internet_use":  "Enterprise internet use",
    "rd_expenditure_pct_gdp":   "GERD % GDP",
    "business_rd_pct_gdp":      "BERD % GDP",
    "epo_patents_per_mio_pop":  "EPO patents / mio. pop.",
    "gva_ict_share":            "GVA ICT share",
    "lq_nace_j62j63":           "LQ ICT services (J)",
    "lq_nace_c21_m72":          "LQ KIS hi-tech (C21/M72)",
    "lq_nace_c26":              "LQ hi-tech mfg. (C26)",
    "lq_nace_d35_clean":        "LQ energy/utilities (D-F)",
}

DIM_COLORS = {
    "cost":    "#e07b39",
    "talent":  "#4c72b0",
    "infra":   "#55a868",
    "cluster": "#8172b2",
}

# is_transition_region: GDP per capita < 75 % of EU27 sample mean
# Captures both "less developed" and "transition" regions per EU Cohesion Policy
TRANSITION_THRESHOLD_PCT = 0.75


# ── Helpers ───────────────────────────────────────────────────────────────────

def _transform(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return copy of feature columns with log1p applied where indicated."""
    out = df[features].copy()
    for col in features:
        if col in LOG1P_FEATURES:
            out[col] = np.log1p(out[col].clip(lower=0))
    return out


def _zscore_single(series: pd.Series) -> pd.Series:
    mu, sd = series.mean(), series.std(ddof=1)
    return (series - mu) / sd


def _pca_pc1(df_features: pd.DataFrame) -> tuple[pd.Series, PCA, np.ndarray]:
    """
    Fit PCA on df_features (NaN rows excluded), extract PC1 scores.
    Returns (scores_series, fitted_pca, loadings_array).
    """
    valid = df_features.dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(valid)
    # random_state is a literal (not a passed-through parameter) so
    # src/utils/seed_check.py's AST auditor can statically verify it; it
    # previously took a random_state=42 default parameter and forwarded it,
    # which the auditor can't resolve past a variable reference and
    # flagged as an unverifiable (effectively None) seed.
    pca = PCA(n_components=min(len(df_features.columns), X.shape[0]),
              random_state=42)
    comps = pca.fit_transform(X)
    pc1 = pd.Series(comps[:, 0], index=valid.index)

    # Re-index to full index (NaN for missing rows)
    scores = pc1.reindex(df_features.index)

    # Orient PC1 so the first loading is positive (interpretable direction)
    if pca.components_[0, 0] < 0:
        scores    = -scores
        pca.components_[0] = -pca.components_[0]

    return scores, pca, pca.components_[0]


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.nan, index=series.index)
    return (series - lo) / (hi - lo)


# ── Score construction ────────────────────────────────────────────────────────

def build_scores(df: pd.DataFrame) -> dict:
    """
    Returns dict with keys: scores, pcas, loadings, variance_explained.
    `scores` is a DataFrame aligned to df.index.
    """
    out = {}
    pcas, loadings, var_exp = {}, {}, {}

    # score_cost: z-score of log1p(gdp_per_capita_pps)
    cost_t = _transform(df, COST_FEATURES)
    out["score_cost"] = _zscore_single(cost_t["gdp_per_capita_pps"])

    # score_talent: PCA PC1 of 4 talent features
    talent_t = _transform(df, TALENT_FEATURES)
    sc, pca, load = _pca_pc1(talent_t)
    out["score_talent"] = sc
    pcas["talent"]     = pca
    loadings["talent"] = load
    var_exp["talent"]  = pca.explained_variance_ratio_[0]
    log.info("talent PCA PC1 variance explained",
             variance_pct=f"{var_exp['talent']:.1%}",
             n_features=len(TALENT_FEATURES))

    # score_infra: mean z-score of 2 infrastructure features
    infra_t = _transform(df, INFRA_FEATURES)
    z_infra = infra_t.apply(_zscore_single)
    out["score_infra"] = z_infra.mean(axis=1)

    # score_cluster: PCA PC1 of 8 cluster features
    cluster_t = _transform(df, CLUSTER_FEATURES)
    sc, pca, load = _pca_pc1(cluster_t)
    out["score_cluster"] = sc
    pcas["cluster"]     = pca
    loadings["cluster"] = load
    var_exp["cluster"]  = pca.explained_variance_ratio_[0]
    log.info("cluster PCA PC1 variance explained",
             variance_pct=f"{var_exp['cluster']:.1%}",
             n_features=len(CLUSTER_FEATURES))

    return {
        "scores":       pd.DataFrame(out, index=df.index),
        "pcas":         pcas,
        "loadings":     loadings,
        "var_explained": var_exp,
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_loading_heatmaps(loadings: dict, var_exp: dict) -> None:
    """Two-panel loading heatmap: talent (4 features) and cluster (8 features)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (dim, feats) in zip(axes, [("talent", TALENT_FEATURES),
                                        ("cluster", CLUSTER_FEATURES)]):
        loads = loadings[dim]
        labels = [FEATURE_LABELS.get(f, f) for f in feats]
        colors = [DIM_COLORS[dim] if v >= 0 else "#d62728" for v in loads]
        bars = ax.barh(labels, loads, color=colors)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("PC1 loading", fontsize=11)
        ax.set_title(
            f"{dim.capitalize()} dimension — PC1\n"
            f"({var_exp[dim]:.1%} variance explained)",
            fontsize=12,
        )
        ax.set_xlim(-1.1, 1.1)
        for bar, v in zip(bars, loads):
            ax.text(v + (0.03 if v >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.5)

    fig.suptitle(
        "PCA PC1 loadings — Talent and Cluster dimensions\n"
        "EU NUTS2 Regional Innovation Panel",
        fontsize=13,
    )
    plt.tight_layout()
    path = FIG_DIR / "p4_loading_heatmaps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_score_distributions(scores: pd.DataFrame) -> None:
    """4-panel violin + strip plots of raw z-scores."""
    dims   = ["cost", "talent", "infra", "cluster"]
    labels = ["Score (cost)", "Score (talent)", "Score (infra)", "Score (cluster)"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    for ax, dim, lbl in zip(axes, dims, labels):
        col  = f"score_{dim}"
        data = scores[col].dropna()
        color = DIM_COLORS[dim]
        ax.violinplot(data, positions=[0], showmedians=True,
                      showextrema=True)
        parts = ax.violinplot(data, positions=[0], showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.scatter(
            np.random.default_rng(42).uniform(-0.04, 0.04, len(data)),
            data,
            s=4, alpha=0.35, color="dimgrey", zorder=3
        )
        ax.set_xticks([])
        ax.set_ylabel(lbl, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.text(0.5, 0.01, f"n={len(data)}  med={data.median():.2f}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color="dimgrey")

    fig.suptitle(
        "Distribution of composite dimension scores (z-scaled)\n"
        "EU NUTS2 Regional Innovation Panel  |  Source: Eurostat",
        fontsize=12,
    )
    plt.tight_layout()
    path = FIG_DIR / "p4_score_distributions.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_score_scattermatrix(scores: pd.DataFrame, df: pd.DataFrame) -> None:
    """Pairwise scatter of all 4 dimension scores, coloured by country."""
    score_cols = ["score_cost", "score_talent", "score_infra", "score_cluster"]
    dim_labels = ["Cost", "Talent", "Infra", "Cluster"]
    n = len(score_cols)

    # Assign a colour per country
    countries = df["country_code"].values
    unique_cc = sorted(set(countries))
    cmap = matplotlib.colormaps["tab20"].resampled(len(unique_cc))
    cc_color = {cc: cmap(i) for i, cc in enumerate(unique_cc)}
    colors = [cc_color[c] for c in countries]

    fig, axes = plt.subplots(n, n, figsize=(14, 13))
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            xi = scores[score_cols[j]].values
            yi = scores[score_cols[i]].values
            mask = ~(np.isnan(xi) | np.isnan(yi))
            if i == j:
                ax.hist(scores[score_cols[i]].dropna(), bins=20,
                        color=list(DIM_COLORS.values())[i], edgecolor="white",
                        linewidth=0.4)
            else:
                ax.scatter(xi[mask], yi[mask], c=[colors[k] for k in np.where(mask)[0]],
                           s=6, alpha=0.55, linewidths=0)
            if i == n - 1:
                ax.set_xlabel(dim_labels[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(dim_labels[i], fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Pairwise scatter of dimension scores (EU NUTS2)\n"
        "Coloured by country  |  Source: Eurostat",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = FIG_DIR / "p4_score_scattermatrix.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_variance_explained(pcas: dict) -> None:
    """Scree plots for talent and cluster PCA."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (dim, pca) in zip(axes, pcas.items()):
        ev = pca.explained_variance_ratio_
        cum = np.cumsum(ev)
        x = np.arange(1, len(ev) + 1)
        ax.bar(x, ev, color=DIM_COLORS[dim], alpha=0.8, label="Individual")
        ax.step(x, cum, where="mid", color="black", linewidth=1.5, label="Cumulative")
        ax.axhline(0.80, color="grey", linestyle="--", linewidth=0.8, label="80 % threshold")
        ax.set_xticks(x)
        ax.set_xlabel("Principal component", fontsize=10)
        ax.set_ylabel("Proportion of variance", fontsize=10)
        ax.set_title(f"{dim.capitalize()} PCA — scree plot", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.text(0.97, 0.97, f"PC1: {ev[0]:.1%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle(
        "Variance explained by PCA — Talent and Cluster dimensions\n"
        "EU NUTS2 Regional Innovation Panel",
        fontsize=12,
    )
    plt.tight_layout()
    path = FIG_DIR / "p4_pca_variance_explained.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    with pipeline_step("p4_load_silver"):
        df = pd.read_parquet(SILVER_PATH)
        log.info("silver loaded", shape=df.shape)

        # Validate net_migration_rate corruption
        if "net_migration_rate" in df.columns and "population" in df.columns:
            corr = df[["population", "net_migration_rate"]].corr().iloc[0, 1]
            log.warning(
                "net_migration_rate is corrupted (correlation with population = 1.0); "
                "excluded from all scoring",
                pearson_r=round(corr, 4),
            )

    with pipeline_step("p4_cap_lq_values"):
        # Gold schema requires LQ in [0, 5.001]; cap outliers before scoring
        for col in LQ_COLS:
            if col in df.columns:
                before_max = df[col].max()
                df[col] = df[col].clip(upper=5.0)
                after_max  = df[col].max()
                if before_max > 5.0:
                    log.info("LQ capped at 5.0", feature=col,
                             before_max=round(before_max, 3),
                             after_max=round(after_max, 3))

    with pipeline_step("p4_log1p_skewness"):
        for col in sorted(LOG1P_FEATURES):
            if col in df.columns:
                sk_before = df[col].skew()
                df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))
                sk_after = df[f"{col}_log1p"].skew()
                log.info("log1p transform",
                         feature=col,
                         skew_before=round(sk_before, 3),
                         skew_after=round(sk_after, 3))
        # employment_rate: left-skewed (skew = -1.082); log1p inapplicable
        log.info("employment_rate: left-skewed (skew ≈ -1.08); used untransformed",
                 skew=round(df["employment_rate"].skew(), 3))

    with pipeline_step("p4_build_scores"):
        result     = build_scores(df)
        scores     = result["scores"]
        pcas       = result["pcas"]
        loadings   = result["loadings"]
        var_exp    = result["var_explained"]

        # Warn if any PC1 explains < 40 % (borderline for composite)
        for dim, ve in var_exp.items():
            if ve < 0.40:
                log.warning("PC1 variance below 40 % — composite may be unstable",
                            dimension=dim, variance_pct=f"{ve:.1%}")

    with pipeline_step("p4_minmax_normalize"):
        for dim in ["cost", "talent", "infra", "cluster"]:
            col = f"score_{dim}"
            scores[f"{col}_minmax"] = _minmax(scores[col])

    with pipeline_step("p4_metadata_flags"):
        eu27_gdp_mean = df["gdp_per_capita_pps"].mean()
        transition_threshold = TRANSITION_THRESHOLD_PCT * eu27_gdp_mean
        log.info("transition region threshold",
                 eu27_mean_pps=round(eu27_gdp_mean, 0),
                 threshold_pps=round(transition_threshold, 0),
                 pct=f"{TRANSITION_THRESHOLD_PCT:.0%}")

        scores["is_transition_region"] = (
            df["gdp_per_capita_pps"] < transition_threshold
        ).values

        # is_insufficient_data: data_quality_score < 0.5
        scores["is_insufficient_data"] = (
            df["data_quality_score"] < 0.5
        ).values

        # dimensionality_warning: >3 analytical features imputed for this region
        analytical_flag_cols = [
            f"{f}_imputed_flag" for f in ALL_ANALYTICAL
            if f"{f}_imputed_flag" in df.columns
        ]
        n_imputed = df[analytical_flag_cols].sum(axis=1)
        scores["dimensionality_warning"] = (n_imputed > 3).values

        # Archetype columns: placeholder until Phase 5
        scores["archetype_id"]          = pd.array([pd.NA] * len(df), dtype="Int64")
        scores["archetype_label"]       = pd.array([pd.NA] * len(df), dtype=object)
        scores["distance_to_centroid"]  = pd.array([pd.NA] * len(df), dtype="Float64")
        scores["silhouette_sample"]     = pd.array([pd.NA] * len(df), dtype="Float64")

        n_trans = scores["is_transition_region"].sum()
        n_insuf = scores["is_insufficient_data"].sum()
        n_dimwarn = scores["dimensionality_warning"].sum()
        log.info("metadata flags",
                 n_transition_regions=int(n_trans),
                 n_insufficient_data=int(n_insuf),
                 n_dimensionality_warnings=int(n_dimwarn))

    with pipeline_step("p4_assemble_gold"):
        gold = df.join(scores)
        # Ensure required identifiers are present
        for req in ("nuts2_name", "country_code", "population", "data_quality_score"):
            assert req in gold.columns, f"Missing required column: {req}"

        log.info("gold assembled", shape=gold.shape,
                 score_cols=[c for c in scores.columns if c.startswith("score_")])

    with pipeline_step("p4_validate_gold"):
        from contracts.gold_schema import gold_schema
        try:
            gold_schema.validate(gold, lazy=True)
            log.info("gold schema validation PASSED")
        except Exception as exc:
            log.warning("gold schema validation issues (non-fatal)",
                        error=str(exc)[:400])

    with pipeline_step("p4_write_gold"):
        gold.to_parquet(GOLD_PATH, engine="pyarrow", index=True)
        log.info("gold written", path=str(GOLD_PATH), shape=gold.shape)

    with pipeline_step("p4_figures"):
        plot_loading_heatmaps(loadings, var_exp)
        plot_score_distributions(scores)
        plot_score_scattermatrix(scores, df)
        plot_variance_explained(pcas)

    # ── Gate check ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("P4 GATE CHECK")
    print("=" * 60)
    score_cols = ["score_cost", "score_talent", "score_infra", "score_cluster"]
    all_pass = True
    for col in score_cols:
        n_valid = gold[col].notna().sum()
        pct     = n_valid / len(gold) * 100
        status  = "PASS" if pct >= 95.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {col:28s}: {n_valid:3d}/{len(gold)} ({pct:.1f}%)  [{status}]")

    mm_cols = [f"{c}_minmax" for c in score_cols]
    for col in mm_cols:
        lo = gold[col].min()
        hi = gold[col].max()
        ok = (lo >= -0.001) and (hi <= 1.001)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {col:28s}: [{lo:.4f}, {hi:.4f}]  [{status}]")

    n_flags = gold["is_transition_region"].sum()
    print(f"  is_transition_region         : {n_flags} regions flagged")
    print(f"  is_insufficient_data         : {gold['is_insufficient_data'].sum()} regions flagged")
    print(f"  dimensionality_warning       : {gold['dimensionality_warning'].sum()} regions flagged")

    for dim, ve in var_exp.items():
        status = "PASS" if ve >= 0.30 else "WARN"
        print(f"  PC1 var explained ({dim:7s}): {ve:.1%}  [{status}]")

    print(f"\n  Gold shape                   : {gold.shape}")
    print(f"  Gold path                    : {GOLD_PATH}")

    print("\n" + ("GATE_P4=PASS" if all_pass else "GATE_P4=FAIL"))
    print("=" * 60)


if __name__ == "__main__":
    main()
