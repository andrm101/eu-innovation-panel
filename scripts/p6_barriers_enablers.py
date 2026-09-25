"""
Phase 6 — Barriers & Enablers Taxonomy
Characterises which features are most strongly *associated with* archetype
membership (non-causal, observational, cross-sectional). All language is
descriptive: "associated with", "correlates with", "patterns consistent with".

Analyses:
  1. Cohen's d  (effect size) with bootstrap 95 % CI for each feature.
  2. Mann-Whitney U (non-parametric) + Benjamini-Hochberg FDR correction.
  3. Multivariate logistic regression (statsmodels) on standardised features
     — odds ratios, 95 % CI, pseudo-R², VIF check.
  4. Within-archetype coefficient of variation — identifies features with
     high internal heterogeneity that the two-group solution masks.

Excluded features:
  - net_migration_rate: corrupted (= population values in Silver)
  - population: context-only, not analytically scored
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import structlog
import scipy.stats as sps
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.logging_config import configure_logging, pipeline_step

configure_logging()
log = structlog.get_logger(__name__)

GOLD_PATH = ROOT / "data" / "gold" / "region_profiles_gold.parquet"
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Feature catalogue ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "gdp_per_capita_pps",
    "hrst_per_1000",
    "tertiary_enrolment_rate",
    "employment_rate",
    "hi_tech_employment_pct",
    "broadband_penetration_pct",
    "enterprise_internet_use",
    "population_density",
    "rd_expenditure_pct_gdp",
    "business_rd_pct_gdp",
    "epo_patents_per_mio_pop",
    "gva_ict_share",
    "lq_nace_j62j63",
    "lq_nace_c21_m72",
    "lq_nace_c26",
    "lq_nace_d35_clean",
]

FEATURE_LABELS = {
    "gdp_per_capita_pps":        "GDP/capita (PPS)",
    "hrst_per_1000":             "HRST per 1 000 pop.",
    "tertiary_enrolment_rate":   "Tertiary attainment rate",
    "employment_rate":           "Employment rate",
    "hi_tech_employment_pct":    "Hi-tech employment %",
    "broadband_penetration_pct": "Broadband penetration",
    "enterprise_internet_use":   "Enterprise internet use",
    "population_density":        "Population density",
    "rd_expenditure_pct_gdp":    "GERD % GDP",
    "business_rd_pct_gdp":       "BERD % GDP",
    "epo_patents_per_mio_pop":   "EPO patents / mio. pop.",
    "gva_ict_share":             "GVA ICT share",
    "lq_nace_j62j63":            "LQ ICT services (J)",
    "lq_nace_c21_m72":           "LQ KIS hi-tech (C21/M72)",
    "lq_nace_c26":               "LQ hi-tech mfg. (C26)",
    "lq_nace_d35_clean":         "LQ energy/utilities (D-F)",
}

DIMENSION_MAP = {
    "gdp_per_capita_pps":        "cost",
    "hrst_per_1000":             "talent",
    "tertiary_enrolment_rate":   "talent",
    "employment_rate":           "talent",
    "hi_tech_employment_pct":    "talent",
    "broadband_penetration_pct": "infra",
    "enterprise_internet_use":   "infra",
    "population_density":        "context",
    "rd_expenditure_pct_gdp":    "cluster",
    "business_rd_pct_gdp":       "cluster",
    "epo_patents_per_mio_pop":   "cluster",
    "gva_ict_share":             "cluster",
    "lq_nace_j62j63":            "cluster",
    "lq_nace_c21_m72":           "cluster",
    "lq_nace_c26":               "cluster",
    "lq_nace_d35_clean":         "cluster",
}

DIM_COLORS = {
    "cost":    "#e07b39",
    "talent":  "#4c72b0",
    "infra":   "#55a868",
    "cluster": "#8172b2",
    "context": "#aaaaaa",
}

LOG1P_FEATURES = {
    "gdp_per_capita_pps", "hi_tech_employment_pct",
    "rd_expenditure_pct_gdp", "business_rd_pct_gdp",
    "epo_patents_per_mio_pop", "gva_ict_share",
    "lq_nace_j62j63", "lq_nace_c21_m72", "lq_nace_c26",
}

ARCHETYPE_LABELS = {0: "Catching-up & peripheral", 1: "High-performance"}
N_BOOTSTRAP = 999


# ── Analysis helpers ──────────────────────────────────────────────────────────

def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to right-skewed features (consistent with Phase 4)."""
    out = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        if col in LOG1P_FEATURES:
            out[col] = np.log1p(out[col].clip(lower=0))
    return out


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d: (mean_y - mean_x) / pooled std."""
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2)
                 / (nx + ny - 2))
    return (y.mean() - x.mean()) / sp if sp > 0 else 0.0


def bootstrap_cohens_d(x: np.ndarray, y: np.ndarray,
                       n_boot: int = N_BOOTSTRAP,
                       rng: np.random.Generator | None = None
                       ) -> tuple[float, float]:
    """Bootstrap 95 % CI for Cohen's d (percentile method)."""
    if rng is None:
        rng = np.random.default_rng(42)
    boots = np.array([
        cohens_d(
            rng.choice(x, size=len(x), replace=True),
            rng.choice(y, size=len(y), replace=True),
        )
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; returns boolean array of rejections."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=int)
    ranked[order] = np.arange(1, n + 1)
    threshold = ranked / n * alpha
    reject = pvals <= threshold
    # BH: if p_i <= (i/n)*alpha for any i, reject all with smaller p
    max_k = np.max(np.where(pvals[order] <= threshold[order], ranked[order], 0))
    return ranked <= max_k


def compute_effect_sizes(df_t: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Cohen's d, Mann-Whitney U, and BH-FDR for each feature."""
    rng = np.random.default_rng(42)
    rows = []
    for col in FEATURE_COLS:
        a0 = df_t.loc[labels == 0, col].dropna().values
        a1 = df_t.loc[labels == 1, col].dropna().values
        d    = cohens_d(a0, a1)
        ci_lo, ci_hi = bootstrap_cohens_d(a0, a1, rng=rng)
        stat, pval = sps.mannwhitneyu(a0, a1, alternative="two-sided")
        rows.append({
            "feature":   col,
            "label":     FEATURE_LABELS[col],
            "dimension": DIMENSION_MAP[col],
            "cohens_d":  d,
            "ci_lo":     ci_lo,
            "ci_hi":     ci_hi,
            "mwu_stat":  stat,
            "mwu_pval":  pval,
            "mean_a0":   a0.mean(),
            "mean_a1":   a1.mean(),
        })
    result = pd.DataFrame(rows).set_index("feature")
    result["fdr_reject"] = bh_fdr(result["mwu_pval"].values)
    result["abs_d"] = result["cohens_d"].abs()
    return result.sort_values("cohens_d", ascending=False)


def logistic_regression(X_std: np.ndarray, y: np.ndarray,
                        feature_names: list[str]) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """
    Fit sklearn L2-regularised logistic regression (C=0.5).
    L2 regularisation handles quasi-complete separation arising from highly
    discriminating features (many |d| > 1.5). Coefficients indicate direction
    and relative association strength; formal inference is not claimed.
    Bootstrap 95 % CIs (n=499) characterise coefficient stability.
    Returns (sklearn_model, coef_df, vif_df).
    """
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = LogisticRegression(C=0.5, solver="lbfgs",
                                   max_iter=1000, random_state=42)
        model.fit(X_std, y)
    coefs = model.coef_[0]

    # Bootstrap 95 % CIs
    rng         = np.random.default_rng(42)
    n_b         = 499
    boot_params = np.zeros((n_b, len(feature_names)))
    for b in range(n_b):
        idx = rng.integers(0, len(y), size=len(y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            m_b = LogisticRegression(C=0.5, solver="lbfgs",
                                     max_iter=1000, random_state=42)
            m_b.fit(X_std[idx], y[idx])
        boot_params[b] = m_b.coef_[0]

    ci_lo = np.percentile(boot_params, 2.5, axis=0)
    ci_hi = np.percentile(boot_params, 97.5, axis=0)

    coef_df = pd.DataFrame({
        "coef":   coefs,
        "ci_lo":  ci_lo,
        "ci_hi":  ci_hi,
        "pval":   np.nan,  # formal p-values not reported; CIs are the inferential summary
    }, index=feature_names)

    # VIF on X_std (statsmodels; no constant column)
    vif_vals = [variance_inflation_factor(X_std, i) for i in range(X_std.shape[1])]
    vif_df   = pd.DataFrame({"feature": feature_names, "vif": vif_vals}).set_index("feature")

    acc = model.score(X_std, y)
    log.info("L2-regularised logistic regression (sklearn, C=0.5)",
             accuracy=round(acc, 4),
             n_obs=len(y),
             n_high_vif=int((vif_df["vif"] > 5).sum()),
             note="coefficients indicate direction/rank; formal p-values suppressed due to separation")

    return model, coef_df, vif_df


def within_archetype_cv(df_t: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Coefficient of variation per feature per archetype (CV = std/|mean|)."""
    rows = []
    for col in FEATURE_COLS:
        for aid in [0, 1]:
            vals = df_t.loc[labels == aid, col].dropna()
            mu   = vals.mean()
            cv   = vals.std(ddof=1) / abs(mu) if abs(mu) > 1e-9 else np.nan
            rows.append({"feature": col, "archetype": aid, "cv": cv})
    return (pd.DataFrame(rows)
            .pivot(index="feature", columns="archetype", values="cv")
            .rename(columns={0: "A1: Catching-up", 1: "A2: High-performance"}))


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_cohens_d(eff: pd.DataFrame) -> None:
    """Horizontal bar chart: Cohen's d with 95 % bootstrap CI."""
    n = len(eff)
    fig, ax = plt.subplots(figsize=(10, 0.55 * n + 2))

    eff_sorted = eff.sort_values("cohens_d")
    colors = [DIM_COLORS[eff_sorted.loc[f, "dimension"]] for f in eff_sorted.index]

    for i, (feat, row) in enumerate(eff_sorted.iterrows()):
        col = DIM_COLORS[row["dimension"]]
        ax.barh(i, row["cohens_d"], color=col, alpha=0.85, height=0.6)
        ax.errorbar(row["cohens_d"], i,
                    xerr=[[row["cohens_d"] - row["ci_lo"]],
                          [row["ci_hi"] - row["cohens_d"]]],
                    fmt="none", color="black", capsize=3, linewidth=1)
        sig = "*" if row["fdr_reject"] else ""
        ax.text(
            row["cohens_d"] + (0.05 if row["cohens_d"] >= 0 else -0.05),
            i,
            f"{row['cohens_d']:.2f}{sig}",
            va="center",
            ha="left" if row["cohens_d"] >= 0 else "right",
            fontsize=7.5,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [FEATURE_LABELS[f] for f in eff_sorted.index], fontsize=8.5
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.8,  color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axvline(-0.8, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axvline(0.5,  color="grey", linestyle=":",  linewidth=0.7, alpha=0.4)
    ax.axvline(-0.5, color="grey", linestyle=":",  linewidth=0.7, alpha=0.4)
    ax.set_xlabel("Cohen's d  (A2 high-performance − A1 catching-up)\n"
                  "Error bars: 95 % bootstrap CI  |  * FDR-significant (BH, a=0.05)",
                  fontsize=9)
    ax.set_title(
        "Feature associations with archetype membership  |  EU NUTS2\n"
        "Positive d: feature correlates with high-performance archetype  |  Source: Eurostat",
        fontsize=11,
    )

    # Dimension legend
    patches = [mpatches.Patch(color=DIM_COLORS[d], label=d.capitalize())
               for d in ["cost", "talent", "infra", "cluster", "context"]]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    path = FIG_DIR / "p6_cohens_d.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_logistic_coef(coef_df: pd.DataFrame, vif_df: pd.DataFrame) -> None:
    """Coefficient plot from multivariate logistic regression (log-odds scale)."""
    df = coef_df.copy()
    df["label"]    = [FEATURE_LABELS.get(f, f) for f in df.index]
    df["dim_color"]= [DIM_COLORS[DIMENSION_MAP.get(f, "context")] for f in df.index]
    df = df.sort_values("coef")

    n = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * n + 2.5),
                             gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    for i, (feat, row) in enumerate(df.iterrows()):
        ax.barh(i, row["coef"], color=row["dim_color"], alpha=0.8, height=0.6)
        ax.errorbar(row["coef"], i,
                    xerr=[[row["coef"] - row["ci_lo"]],
                          [row["ci_hi"] - row["coef"]]],
                    fmt="none", color="black", capsize=3, linewidth=1)
        sig = "**" if row["pval"] < 0.01 else ("*" if row["pval"] < 0.05 else "")
        ax.text(row["coef"] + (0.05 if row["coef"] >= 0 else -0.05), i,
                f"{row['coef']:.2f}{sig}",
                va="center", ha="left" if row["coef"] >= 0 else "right",
                fontsize=7.5)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["label"].tolist(), fontsize=8.5)
    ax.set_xlabel("Log-odds coefficient (standardised features)\n"
                  "95 % CI shown  |  * p<0.05  ** p<0.01  (multivariate logistic regression)",
                  fontsize=9)
    ax.set_title(
        "Multivariate logistic regression coefficients\n"
        "Outcome: A2 high-performance archetype (vs A1 catching-up)",
        fontsize=10,
    )

    # VIF panel
    ax2 = axes[1]
    vif_order = [f for f in df.index if f in vif_df.index]
    vif_vals  = [vif_df.loc[f, "vif"] for f in vif_order]
    vif_colors = ["#c44e52" if v > 5 else "#55a868" for v in vif_vals]
    ax2.barh(range(n), vif_vals, color=vif_colors, alpha=0.8)
    ax2.axvline(5, color="grey", linestyle="--", linewidth=0.8, label="VIF=5")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([])
    ax2.set_xlabel("VIF", fontsize=9)
    ax2.set_title("VIF\n(red > 5)", fontsize=10)
    ax2.legend(fontsize=8)

    plt.suptitle(
        "Multivariate logistic regression — feature associations with archetype\n"
        "EU NUTS2 Regional Innovation Panel  |  Source: Eurostat",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    path = FIG_DIR / "p6_logistic_coef.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_within_variance(cv_df: pd.DataFrame) -> None:
    """Heatmap of coefficient of variation per feature per archetype."""
    ordered = cv_df.reindex(FEATURE_COLS)
    row_labels = [FEATURE_LABELS.get(f, f) for f in ordered.index]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(ordered) + 2))
    im = ax.imshow(ordered.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=ordered.values[np.isfinite(ordered.values)].max())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(ordered.columns.tolist(), fontsize=10)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    for i in range(len(ordered)):
        for j in range(2):
            v = ordered.iloc[i, j]
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "N/A",
                    ha="center", va="center", fontsize=8.5)
    plt.colorbar(im, ax=ax, shrink=0.8,
                 label="Coefficient of variation (std / |mean|)")
    ax.set_title(
        "Within-archetype feature heterogeneity\n"
        "Higher CV → greater internal variation masked by archetype label\n"
        "EU NUTS2 Regional Innovation Panel  |  Source: Eurostat",
        fontsize=10,
    )
    plt.tight_layout()
    path = FIG_DIR / "p6_within_variance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_feature_violin(df: pd.DataFrame, eff: pd.DataFrame) -> None:
    """Violin plots of top-8 discriminating features by archetype."""
    top8 = eff.sort_values("abs_d", ascending=False).head(8).index.tolist()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for ax, feat in zip(axes.flatten(), top8):
        a0_vals = df.loc[df["archetype_id"] == 0, feat].dropna()
        a1_vals = df.loc[df["archetype_id"] == 1, feat].dropna()
        data    = [a0_vals.values, a1_vals.values]
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True)
        parts["bodies"][0].set_facecolor("#4c72b0")
        parts["bodies"][1].set_facecolor("#dd8452")
        for body in parts["bodies"]:
            body.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["A1: Catching-up", "A2: High-perf."], fontsize=8)
        d_val = eff.loc[feat, "cohens_d"]
        ax.set_title(f"{FEATURE_LABELS[feat]}\nd = {d_val:.2f}", fontsize=8.5)
        ax.set_ylabel("log1p(value)" if feat in LOG1P_FEATURES else "value", fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Distribution of top-8 discriminating features by archetype  |  EU NUTS2\n"
        "Ranked by |Cohen's d|  |  Source: Eurostat",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p6_feature_violin.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    with pipeline_step("p6_load_gold"):
        df = pd.read_parquet(GOLD_PATH)
        log.info("gold loaded", shape=df.shape,
                 a0=int((df["archetype_id"] == 0).sum()),
                 a1=int((df["archetype_id"] == 1).sum()))

    with pipeline_step("p6_transform_features"):
        df_t   = apply_transforms(df)
        labels = df["archetype_id"].values.astype(int)
        log.info("features transformed", log1p_count=len(LOG1P_FEATURES))

    with pipeline_step("p6_effect_sizes"):
        eff = compute_effect_sizes(df_t, labels)
        n_sig  = int(eff["fdr_reject"].sum())
        n_large = int((eff["abs_d"] >= 0.8).sum())
        log.info("effect sizes",
                 n_fdr_significant=n_sig,
                 n_large_effect=n_large,
                 n_features=len(eff))
        for feat, row in eff.iterrows():
            log.info("feature effect",
                     feature=feat,
                     cohens_d=round(row["cohens_d"], 3),
                     ci_95=f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
                     fdr_significant=bool(row["fdr_reject"]))

    with pipeline_step("p6_logistic_regression"):
        scaler  = StandardScaler()
        X_std   = scaler.fit_transform(df_t[FEATURE_COLS].values)
        lr_res, coef_df, vif_df = logistic_regression(X_std, labels, FEATURE_COLS)

        n_high_vif = int((vif_df["vif"] > 5).sum())
        if n_high_vif > 0:
            log.warning("high VIF features detected — multivariate coefficients "
                        "should be interpreted with caution",
                        n_high_vif=n_high_vif,
                        features=vif_df[vif_df["vif"] > 5].index.tolist())

        log.info("logistic regression summary",
                 accuracy=round(lr_res.score(X_std, labels), 4),
                 note="L2-regularised sklearn (C=0.5); coefficients indicate direction not marginal effects")

    with pipeline_step("p6_within_variance"):
        cv_df = within_archetype_cv(df_t, labels)
        high_cv = cv_df[cv_df.max(axis=1) > 0.5]
        log.info("within-archetype CV",
                 n_high_cv_features=len(high_cv),
                 features=high_cv.index.tolist())

    with pipeline_step("p6_save_results"):
        out_dir = ROOT / "analysis"
        out_dir.mkdir(exist_ok=True)

        eff_out = eff.reset_index().rename(columns={"feature": "feature_code"})
        eff_out.to_csv(out_dir / "p6_effect_sizes.csv", index=False)
        log.info("effect sizes saved", path=str(out_dir / "p6_effect_sizes.csv"))

        coef_out = coef_df.reset_index().rename(columns={"index": "feature_code"})
        coef_out.to_csv(out_dir / "p6_logistic_coefs.csv", index=False)
        log.info("logistic coefficients saved",
                 path=str(out_dir / "p6_logistic_coefs.csv"))

    with pipeline_step("p6_figures"):
        plot_cohens_d(eff)
        plot_logistic_coef(coef_df, vif_df)
        plot_within_variance(cv_df)
        plot_feature_violin(df, eff)

    # ── Gate check ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("P6 GATE CHECK")
    print("=" * 65)

    print(f"  Features analysed              : {len(eff)}")
    print(f"  FDR-significant (BH, a=0.05)   : {n_sig}/{len(eff)}")
    print(f"  Large-effect features (|d|>=0.8): {n_large}")
    print(f"  Logistic accuracy (L2, C=0.5)  : {lr_res.score(X_std, labels):.4f}")
    print(f"  VIF > 5                        : {n_high_vif} features")

    print("\n  Top-5 features by |Cohen's d|:")
    for feat, row in eff.sort_values("abs_d", ascending=False).head(5).iterrows():
        print(f"    {FEATURE_LABELS[feat]:40s}: d={row['cohens_d']:+.3f}  "
              f"{'*' if row['fdr_reject'] else ' '}")

    print("\n  Within-archetype high-CV features (CV > 0.5 in either archetype):")
    for feat in high_cv.index:
        print(f"    {FEATURE_LABELS[feat]:40s}: "
              f"CV(A1)={cv_df.loc[feat,'A1: Catching-up']:.3f}  "
              f"CV(A2)={cv_df.loc[feat,'A2: High-performance']:.3f}")

    all_pass = (n_sig >= 10)
    print(f"\nGATE_P6={'PASS' if all_pass else 'WARN'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
