"""
Phase 6c -- Country Fixed-Effects Analysis
Tests whether archetype score separation and feature associations persist
after removing country-level mean differences. Addresses the concern that
archetype differences may simply reflect cross-country variation rather than
genuine within-country regional heterogeneity.

Approach:
  1. Country-demean all 4 dimension scores and 16 features.
  2. Re-compute Cohen's d (with 999-bootstrap 95% CI) on demeaned values.
  3. Logistic regression: archetype_id ~ demeaned features (country FE implicit).
  4. Within-country separation: for mixed countries (both archetypes present),
     report mean within-country silhouette and archetype composition.

All language is descriptive and non-causal.

Outputs:
  analysis/p6c_demeaned_cohens_d.csv   -- Cohen's d before and after demeaning
  analysis/p6c_within_country.csv      -- Per-country archetype composition
  figures/p6c_demeaning_comparison.png -- Scatter: raw d vs demeaned d
  figures/p6c_within_country_bar.png   -- Within-country archetype composition
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import structlog

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils.logging_config import configure_logging, pipeline_step

configure_logging()
log = structlog.get_logger(__name__)

GOLD_PATH = ROOT / "data" / "gold" / "region_profiles_gold.parquet"
ANA_DIR   = ROOT / "analysis"
FIG_DIR   = ROOT / "figures"
ANA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

N_BOOT = 999

SCORE_COLS = ["score_cost", "score_talent", "score_infra", "score_cluster"]
FEATURE_COLS = [
    "gdp_per_capita_pps", "hrst_per_1000", "tertiary_enrolment_rate",
    "employment_rate", "hi_tech_employment_pct", "broadband_penetration_pct",
    "enterprise_internet_use", "population_density", "rd_expenditure_pct_gdp",
    "business_rd_pct_gdp", "epo_patents_per_mio_pop", "gva_ict_share",
    "lq_nace_j62j63", "lq_nace_c21_m72", "lq_nace_c26", "lq_nace_d35_clean",
]
FEATURE_LABELS = {
    "gdp_per_capita_pps":        "GDP/capita (PPS)",
    "hrst_per_1000":             "HRST per 1 000",
    "tertiary_enrolment_rate":   "Tertiary attainment",
    "employment_rate":           "Employment rate",
    "hi_tech_employment_pct":    "Hi-tech employment %",
    "broadband_penetration_pct": "Broadband penetration",
    "enterprise_internet_use":   "Enterprise internet use",
    "population_density":        "Population density",
    "rd_expenditure_pct_gdp":    "GERD % GDP",
    "business_rd_pct_gdp":       "BERD % GDP",
    "epo_patents_per_mio_pop":   "EPO patents / mio.",
    "gva_ict_share":             "GVA ICT share",
    "lq_nace_j62j63":            "LQ ICT services (J)",
    "lq_nace_c21_m72":           "LQ KIS hi-tech (C21/M72)",
    "lq_nace_c26":               "LQ hi-tech mfg. (C26)",
    "lq_nace_d35_clean":         "LQ energy/utilities (D-F)",
}
SCORE_LABELS = {
    "score_cost":    "Prosperity score",
    "score_talent":  "Talent score",
    "score_infra":   "Digital infra. score",
    "score_cluster": "Innovation cluster score",
}

ALL_VARS = SCORE_COLS + FEATURE_COLS
ALL_LABELS = {**SCORE_LABELS, **FEATURE_LABELS}


# ── Statistical helpers ───────────────────────────────────────────────────────

def cohens_d_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT
                ) -> tuple[float, float, float]:
    """Return (d, ci_lo, ci_hi). Positive = b > a."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return (np.nan, np.nan, np.nan)
    pooled_sd = np.sqrt(
        ((n_a - 1) * a.std(ddof=1)**2 + (n_b - 1) * b.std(ddof=1)**2)
        / (n_a + n_b - 2)
    )
    if pooled_sd < 1e-12:
        return (0.0, 0.0, 0.0)
    d = (b.mean() - a.mean()) / pooled_sd
    rng = np.random.default_rng(42)
    boot_d = np.array([
        (rng.choice(b, len(b), replace=True).mean() -
         rng.choice(a, len(a), replace=True).mean()) / pooled_sd
        for _ in range(n_boot)
    ])
    return (d,
            float(np.percentile(boot_d, 2.5)),
            float(np.percentile(boot_d, 97.5)))


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_demeaning_comparison(raw_df: pd.DataFrame) -> None:
    """Scatter: raw Cohen's d vs country-demeaned Cohen's d for all variables."""
    fig, ax = plt.subplots(figsize=(7, 6))

    score_mask   = raw_df["variable"].isin(SCORE_COLS)
    feature_mask = ~score_mask

    ax.scatter(raw_df.loc[feature_mask, "cohens_d_raw"],
               raw_df.loc[feature_mask, "cohens_d_demeaned"],
               s=40, alpha=0.75, color="#4c72b0", label="Feature", zorder=3)
    ax.scatter(raw_df.loc[score_mask, "cohens_d_raw"],
               raw_df.loc[score_mask, "cohens_d_demeaned"],
               s=80, alpha=0.9, color="#dd8452", marker="D",
               label="Dimension score", zorder=4)

    # 45-degree reference line (d unchanged by demeaning)
    lim_min = min(raw_df["cohens_d_raw"].min(), raw_df["cohens_d_demeaned"].min()) - 0.1
    lim_max = max(raw_df["cohens_d_raw"].max(), raw_df["cohens_d_demeaned"].max()) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="grey", linewidth=0.9, linestyle="--", label="No change")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    # Label the score points
    for _, row in raw_df[score_mask].iterrows():
        ax.annotate(SCORE_LABELS.get(row["variable"], row["variable"]).replace(" score", ""),
                    xy=(row["cohens_d_raw"], row["cohens_d_demeaned"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7.5, color="#dd8452")

    ax.set_xlabel("Cohen's d (raw)", fontsize=10)
    ax.set_ylabel("Cohen's d (country-demeaned)", fontsize=10)
    ax.set_title(
        "Effect of country fixed effects on archetype separation\n"
        "(Points below the diagonal: country removal reduces effect size)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    path = FIG_DIR / "p6c_demeaning_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("saved figure", path=str(path))


def plot_within_country_bar(wc_df: pd.DataFrame) -> None:
    """Horizontal bar: A2 share within each country, sorted, mixed countries only."""
    mixed = wc_df[wc_df["n_archetypes"] == 2].copy()
    mixed = mixed.sort_values("a2_share")

    fig, ax = plt.subplots(figsize=(8, max(4, len(mixed) * 0.35)))
    colors = ["#dd8452" if v >= 0.5 else "#4c72b0" for v in mixed["a2_share"]]
    ax.barh(mixed["country_name"], mixed["a2_share"], color=colors,
            edgecolor="white", height=0.65)
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of NUTS2 regions in A2 (Advanced innovation)", fontsize=10)
    ax.set_title(
        "Within-country archetype composition — mixed countries only\n"
        "(Countries with regions in both archetypes)",
        fontsize=10,
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    path = FIG_DIR / "p6c_within_country_bar.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("saved figure", path=str(path))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    with pipeline_step("p6c_load"):
        df = pd.read_parquet(GOLD_PATH).reset_index()
        log.info("loaded gold", shape=df.shape)

    # ── Country-demean ────────────────────────────────────────────────────────
    with pipeline_step("p6c_demean"):
        demean_vars = ALL_VARS
        cc_means = df.groupby("country_code")[demean_vars].transform("mean")
        df_dm = df.copy()
        for v in demean_vars:
            df_dm[v + "_dm"] = df[v] - cc_means[v]
        log.info("country-demeaned", n_vars=len(demean_vars))

    # ── Cohen's d: raw vs demeaned ────────────────────────────────────────────
    with pipeline_step("p6c_cohens_d"):
        a1 = df[df["archetype_id"] == 0]
        a2 = df[df["archetype_id"] == 1]
        a1_dm = df_dm[df_dm["archetype_id"] == 0]
        a2_dm = df_dm[df_dm["archetype_id"] == 1]

        rows = []
        for var in ALL_VARS:
            d_raw, ci_lo_raw, ci_hi_raw = cohens_d_ci(
                a1[var].dropna().values, a2[var].dropna().values
            )
            d_dm, ci_lo_dm, ci_hi_dm = cohens_d_ci(
                a1_dm[var + "_dm"].dropna().values,
                a2_dm[var + "_dm"].dropna().values,
            )
            rows.append({
                "variable":         var,
                "label":            ALL_LABELS.get(var, var),
                "cohens_d_raw":     round(d_raw, 4),
                "ci_lo_raw":        round(ci_lo_raw, 4),
                "ci_hi_raw":        round(ci_hi_raw, 4),
                "cohens_d_demeaned": round(d_dm, 4),
                "ci_lo_demeaned":   round(ci_lo_dm, 4),
                "ci_hi_demeaned":   round(ci_hi_dm, 4),
                "d_change":         round(d_dm - d_raw, 4),
                "pct_retained":     round(abs(d_dm) / max(abs(d_raw), 1e-9) * 100, 1),
            })
            log.info("Cohen's d",
                     variable=var, d_raw=round(d_raw, 3), d_dm=round(d_dm, 3))

        cd_df = pd.DataFrame(rows)
        cd_df.to_csv(ANA_DIR / "p6c_demeaned_cohens_d.csv", index=False)
        log.info("saved", path=str(ANA_DIR / "p6c_demeaned_cohens_d.csv"))
        plot_demeaning_comparison(cd_df)

    # ── Logistic regression (demeaned features) ───────────────────────────────
    with pipeline_step("p6c_logistic"):
        dm_feature_cols = [v + "_dm" for v in FEATURE_COLS]
        X = df_dm[dm_feature_cols].fillna(0).values
        y = df_dm["archetype_id"].values

        scaler = StandardScaler()
        X_std  = scaler.fit_transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000,
                                    random_state=42)
            lr.fit(X_std, y)

        acc = accuracy_score(y, lr.predict(X_std))
        coefs = pd.DataFrame({
            "feature":  FEATURE_COLS,
            "label":    [FEATURE_LABELS[f] for f in FEATURE_COLS],
            "coef":     lr.coef_[0].round(4),
        }).sort_values("coef", ascending=False)
        log.info("demeaned logistic regression",
                 accuracy=round(acc, 4), n_features=len(FEATURE_COLS))

    # ── Within-country analysis ───────────────────────────────────────────────
    with pipeline_step("p6c_within_country"):
        COUNTRY_NAMES = {
            "AT": "Austria",    "BE": "Belgium",   "BG": "Bulgaria",  "CY": "Cyprus",
            "CZ": "Czechia",    "DE": "Germany",   "DK": "Denmark",   "EE": "Estonia",
            "EL": "Greece",     "ES": "Spain",     "FI": "Finland",   "FR": "France",
            "HR": "Croatia",    "HU": "Hungary",   "IE": "Ireland",   "IT": "Italy",
            "LT": "Lithuania",  "LU": "Luxembourg","LV": "Latvia",    "MT": "Malta",
            "NL": "Netherlands","PL": "Poland",    "PT": "Portugal",  "RO": "Romania",
            "SE": "Sweden",     "SI": "Slovenia",  "SK": "Slovakia",
        }
        wc_rows = []
        for cc, grp in df.groupby("country_code"):
            n_total = len(grp)
            n_a2    = int((grp["archetype_id"] == 1).sum())
            n_a1    = n_total - n_a2
            n_arch  = int(grp["archetype_id"].nunique())

            # Mean within-country silhouette
            mean_sil = grp["silhouette_sample"].mean() if "silhouette_sample" in grp else np.nan

            # Within-country score separation (Cohen's d on un-demeaned scores)
            score_d = {}
            if n_arch == 2:
                for sc in SCORE_COLS:
                    g_a1 = grp.loc[grp["archetype_id"] == 0, sc].values
                    g_a2 = grp.loc[grp["archetype_id"] == 1, sc].values
                    d, _, _ = cohens_d_ci(g_a1, g_a2, n_boot=199)
                    score_d[sc + "_d"] = round(d, 3)

            row = {
                "country_code":  cc,
                "country_name":  COUNTRY_NAMES.get(cc, cc),
                "n_regions":     n_total,
                "n_a1":          n_a1,
                "n_a2":          n_a2,
                "a2_share":      round(n_a2 / n_total, 3),
                "n_archetypes":  n_arch,
                "mean_silhouette": round(mean_sil, 4) if pd.notna(mean_sil) else None,
            }
            row.update(score_d)
            wc_rows.append(row)

        wc_df = pd.DataFrame(wc_rows).sort_values("a2_share", ascending=False)
        wc_df.to_csv(ANA_DIR / "p6c_within_country.csv", index=False)
        log.info("within-country table saved",
                 n_mixed=int((wc_df["n_archetypes"] == 2).sum()))
        plot_within_country_bar(wc_df)

    # ── Gate check ────────────────────────────────────────────────────────────
    score_rows = cd_df[cd_df["variable"].isin(SCORE_COLS)]
    feat_rows  = cd_df[cd_df["variable"].isin(FEATURE_COLS)]
    n_mixed    = int((wc_df["n_archetypes"] == 2).sum())
    n_homog    = 27 - n_mixed

    print("\n" + "=" * 70)
    print("P6c COUNTRY FIXED-EFFECTS ANALYSIS — GATE CHECK")
    print("=" * 70)

    print(f"\n  Dimension score separation (raw d vs country-demeaned d):\n")
    print(f"  {'Score':<30} {'d_raw':>8} {'d_demeaned':>12} {'% retained':>12}")
    print("  " + "-" * 65)
    for _, r in score_rows.sort_values("cohens_d_raw", ascending=False).iterrows():
        print(f"  {r['label']:<30} {r['cohens_d_raw']:>+8.3f} "
              f"{r['cohens_d_demeaned']:>+12.3f} {r['pct_retained']:>11.1f}%")

    print(f"\n  Top features by demeaned |d| (n={len(feat_rows)}):\n")
    top_dm = feat_rows.reindex(feat_rows["cohens_d_demeaned"].abs().sort_values(ascending=False).index)
    print(f"  {'Feature':<35} {'d_demeaned':>12} {'% retained':>12}")
    print("  " + "-" * 62)
    for _, r in top_dm.head(8).iterrows():
        print(f"  {r['label']:<35} {r['cohens_d_demeaned']:>+12.3f} {r['pct_retained']:>11.1f}%")

    print(f"\n  Demeaned logistic regression accuracy: {acc:.4f}")

    print(f"\n  Country composition:")
    print(f"    Total EU27 countries        : 27")
    print(f"    Homogeneous (1 archetype)   : {n_homog}")
    print(f"    Mixed (both archetypes)     : {n_mixed}")

    print(f"\n  Artifacts:")
    print(f"    analysis/p6c_demeaned_cohens_d.csv")
    print(f"    analysis/p6c_within_country.csv")
    print(f"    figures/p6c_demeaning_comparison.png")
    print(f"    figures/p6c_within_country_bar.png")

    # Pass if score separation is retained at >50% after demeaning
    min_retained = float(score_rows["pct_retained"].min())
    status = "PASS" if min_retained > 50 else "WARN"
    print(f"\n  Min score separation retained after demeaning: {min_retained:.1f}%")
    print(f"\nGATE_P6c={status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
