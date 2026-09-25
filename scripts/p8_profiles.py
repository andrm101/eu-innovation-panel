"""
Phase 8 -- Regional Archetype Profiles
Generates structured profile documents for each archetype and a full
region-level table. All language is descriptive and non-causal.

Outputs:
  analysis/p8_archetype_profiles.md   -- Markdown narrative profiles
  analysis/p8_archetype_profiles.json -- Structured JSON (for manuscript)
  analysis/p8_region_table.csv        -- Full 242-region table
  figures/p8_feature_comparison.png   -- Mean +/- SD per feature per archetype
  figures/p8_score_profile.png        -- Score heatmap: regions x dimensions
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import structlog

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.logging_config import configure_logging, pipeline_step

configure_logging()
log = structlog.get_logger(__name__)

GOLD_PATH = ROOT / "data" / "gold" / "region_profiles_gold.parquet"
P6_EFF    = ROOT / "analysis" / "p6_effect_sizes.csv"
FIG_DIR   = ROOT / "figures"
ANA_DIR   = ROOT / "analysis"
FIG_DIR.mkdir(exist_ok=True)
ANA_DIR.mkdir(exist_ok=True)

SCORE_COLS = ["score_cost", "score_talent", "score_infra", "score_cluster"]
MM_COLS    = ["score_cost_minmax", "score_talent_minmax",
              "score_infra_minmax", "score_cluster_minmax"]
DIM_LABELS = {
    "score_cost":    "Prosperity (cost)",
    "score_talent":  "Talent",
    "score_infra":   "Digital infrastructure",
    "score_cluster": "Innovation cluster",
}

FEATURE_COLS = [
    "gdp_per_capita_pps", "hrst_per_1000", "tertiary_enrolment_rate",
    "employment_rate", "hi_tech_employment_pct", "broadband_penetration_pct",
    "enterprise_internet_use", "population_density", "rd_expenditure_pct_gdp",
    "business_rd_pct_gdp", "epo_patents_per_mio_pop", "gva_ict_share",
    "lq_nace_j62j63", "lq_nace_c21_m72", "lq_nace_c26", "lq_nace_d35_clean",
]
FEATURE_LABELS = {
    "gdp_per_capita_pps":        "GDP per capita (PPS, EUR)",
    "hrst_per_1000":             "HRST per 1,000 population",
    "tertiary_enrolment_rate":   "Tertiary attainment rate (%)",
    "employment_rate":           "Employment rate (%)",
    "hi_tech_employment_pct":    "Hi-tech employment (%)",
    "broadband_penetration_pct": "Broadband penetration (%)",
    "enterprise_internet_use":   "Enterprise internet use (%)",
    "population_density":        "Population density (per km2)",
    "rd_expenditure_pct_gdp":    "GERD (% GDP)",
    "business_rd_pct_gdp":       "BERD (% GDP)",
    "epo_patents_per_mio_pop":   "EPO patents per million pop.",
    "gva_ict_share":             "GVA ICT share (%)",
    "lq_nace_j62j63":            "Location quotient -- ICT services (J)",
    "lq_nace_c21_m72":           "Location quotient -- KIS hi-tech (C21/M72)",
    "lq_nace_c26":               "Location quotient -- hi-tech mfg. (C26)",
    "lq_nace_d35_clean":         "Location quotient -- energy/utilities (D-F)",
}
DIMENSION_MAP = {
    "gdp_per_capita_pps":        "Prosperity",
    "hrst_per_1000":             "Talent",
    "tertiary_enrolment_rate":   "Talent",
    "employment_rate":           "Talent",
    "hi_tech_employment_pct":    "Talent",
    "broadband_penetration_pct": "Infra",
    "enterprise_internet_use":   "Infra",
    "population_density":        "Context",
    "rd_expenditure_pct_gdp":    "Innovation",
    "business_rd_pct_gdp":       "Innovation",
    "epo_patents_per_mio_pop":   "Innovation",
    "gva_ict_share":             "Innovation",
    "lq_nace_j62j63":            "Innovation",
    "lq_nace_c21_m72":           "Innovation",
    "lq_nace_c26":               "Innovation",
    "lq_nace_d35_clean":         "Innovation",
}
DIM_COLORS = {
    "Prosperity": "#e07b39",
    "Talent":     "#4c72b0",
    "Infra":      "#55a868",
    "Innovation": "#8172b2",
    "Context":    "#aaaaaa",
}
ARCHETYPE_COLORS = {0: "#4c72b0", 1: "#dd8452"}

COUNTRY_NAMES = {
    "AT": "Austria",    "BE": "Belgium",   "BG": "Bulgaria",  "CY": "Cyprus",
    "CZ": "Czechia",    "DE": "Germany",   "DK": "Denmark",   "EE": "Estonia",
    "EL": "Greece",     "ES": "Spain",     "FI": "Finland",   "FR": "France",
    "HR": "Croatia",    "HU": "Hungary",   "IE": "Ireland",   "IT": "Italy",
    "LT": "Lithuania",  "LU": "Luxembourg","LV": "Latvia",    "MT": "Malta",
    "NL": "Netherlands","PL": "Poland",    "PT": "Portugal",  "RO": "Romania",
    "SE": "Sweden",     "SI": "Slovenia",  "SK": "Slovakia",
}


# ---- Profile builders -------------------------------------------------------

def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:,.{decimals}f}"


def build_archetype_profile(df: pd.DataFrame, aid: int, eff: pd.DataFrame | None
                             ) -> dict:
    """Return a dict with all profile data for one archetype."""
    sub  = df[df["archetype_id"] == aid].copy()
    eu27 = df.copy()

    label = sub["archetype_label"].iloc[0]
    n     = len(sub)

    # Country distribution
    cc_counts = sub["country_code"].value_counts()
    cc_dominant = cc_counts[cc_counts / len(sub) >= 0.8].index.tolist()
    cc_top5 = [
        f"{COUNTRY_NAMES.get(cc, cc)} ({cnt})"
        for cc, cnt in cc_counts.head(5).items()
    ]

    # Dimension scores
    score_summary = {}
    for sc in SCORE_COLS:
        score_summary[sc] = {
            "mean":       round(float(sub[sc].mean()), 3),
            "sd":         round(float(sub[sc].std(ddof=1)), 3),
            "eu27_mean":  round(float(eu27[sc].mean()), 3),
            "label":      DIM_LABELS[sc],
        }

    # Feature statistics
    feat_stats = {}
    for fc in FEATURE_COLS:
        vals    = sub[fc].dropna()
        eu_vals = eu27[fc].dropna()
        feat_stats[fc] = {
            "label":      FEATURE_LABELS[fc],
            "dimension":  DIMENSION_MAP[fc],
            "mean":       round(float(vals.mean()), 3) if len(vals) else None,
            "sd":         round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else None,
            "eu27_mean":  round(float(eu_vals.mean()), 3) if len(eu_vals) else None,
            "pct_vs_eu27": round(float((vals.mean() / eu_vals.mean() - 1) * 100), 1)
                           if eu_vals.mean() != 0 and len(vals) else None,
        }

    # Most typical regions (smallest distance to centroid)
    typical = (
        sub.nsmallest(5, "distance_to_centroid")
        .reset_index()[["nuts2_code", "nuts2_name", "country_code",
                        "distance_to_centroid", "silhouette_sample"]]
        .copy()
    )
    typical["distance_to_centroid"] = typical["distance_to_centroid"].round(4)
    typical["silhouette_sample"]    = typical["silhouette_sample"].round(4)

    # Most atypical (largest distance to centroid)
    atypical = (
        sub.nlargest(5, "distance_to_centroid")
        .reset_index()[["nuts2_code", "nuts2_name", "country_code",
                        "distance_to_centroid", "silhouette_sample"]]
        .copy()
    )
    atypical["distance_to_centroid"] = atypical["distance_to_centroid"].round(4)
    atypical["silhouette_sample"]    = atypical["silhouette_sample"].round(4)

    # Within-archetype high-CV features
    high_cv = []
    for fc in FEATURE_COLS:
        vals = sub[fc].dropna()
        mu   = vals.mean()
        cv   = vals.std(ddof=1) / abs(mu) if abs(mu) > 1e-9 else np.nan
        if np.isfinite(cv) and cv > 0.5:
            high_cv.append({"feature": fc, "label": FEATURE_LABELS[fc], "cv": round(cv, 3)})

    # Cohen's d from P6 (positive d = higher in A1)
    top_assoc = []
    if eff is not None:
        eff_sorted = eff.sort_values("cohens_d", ascending=(aid == 0)).head(5)
        for _, row in eff_sorted.iterrows():
            d_sign = "higher" if (row["cohens_d"] > 0) == (aid == 1) else "lower"
            top_assoc.append({
                "feature": FEATURE_LABELS.get(row["feature_code"], row["feature_code"]),
                "cohens_d": round(row["cohens_d"], 3),
                "direction": d_sign,
            })

    # Transition regions
    n_trans = int(sub["is_transition_region"].sum())
    n_warn  = int(sub["dimensionality_warning"].sum())
    mean_dq = round(float(sub["data_quality_score"].mean()), 3)
    mean_sil= round(float(sub["silhouette_sample"].mean()), 4)

    return {
        "archetype_id":          aid,
        "archetype_label":       label,
        "n_regions":             n,
        "n_countries":           int(sub["country_code"].nunique()),
        "countries_dominant":    cc_dominant,
        "top5_countries":        cc_top5,
        "score_summary":         score_summary,
        "feature_stats":         feat_stats,
        "typical_regions":       typical.to_dict("records"),
        "atypical_regions":      atypical.to_dict("records"),
        "high_cv_features":      sorted(high_cv, key=lambda x: -x["cv"]),
        "top_associations":      top_assoc,
        "n_transition_regions":  n_trans,
        "pct_transition":        round(n_trans / n * 100, 1),
        "n_dimensionality_warn": n_warn,
        "mean_data_quality":     mean_dq,
        "mean_silhouette":       mean_sil,
    }


def profile_to_markdown(p: dict, eu27_gdp_mean: float) -> str:
    """Render one archetype profile as a Markdown section."""
    lines = []
    aid   = p["archetype_id"]
    lbl   = p["archetype_label"]

    lines.append(f"## Archetype {aid + 1}: {lbl}\n")
    lines.append(
        f"**{p['n_regions']} NUTS2 regions** across {p['n_countries']} EU member states.\n"
    )

    # Score profile
    lines.append("### Dimension score profile\n")
    lines.append("| Dimension | Archetype mean | EU27 mean | Difference |")
    lines.append("|-----------|---------------|-----------|------------|")
    for sc, s in p["score_summary"].items():
        diff = s["mean"] - s["eu27_mean"]
        sign = "+" if diff >= 0 else ""
        lines.append(
            f"| {s['label']} | {s['mean']:.3f} | {s['eu27_mean']:.3f} | {sign}{diff:.3f} |"
        )
    lines.append("")

    # Transition regions
    lines.append(
        f"**Transition regions** (GDP per capita < 75% EU27 sample mean): "
        f"{p['n_transition_regions']} of {p['n_regions']} "
        f"({p['pct_transition']:.1f}%).\n"
    )

    # Country distribution
    lines.append("### Geographic distribution\n")
    lines.append(
        "Regions are distributed across the following countries "
        "(top five by region count): " + ", ".join(p["top5_countries"]) + ".\n"
    )
    if p["countries_dominant"]:
        dom = [COUNTRY_NAMES.get(c, c) for c in p["countries_dominant"]]
        lines.append(
            f"Countries where this archetype represents >= 80% of national NUTS2 regions: "
            + ", ".join(dom) + ".\n"
        )

    # Key feature associations
    lines.append("### Key feature associations (descriptive)\n")
    lines.append(
        "The following features show the strongest associations with archetype membership "
        "(ordered by |Cohen's d|; observational data, no causal claims):\n"
    )
    lines.append("| Feature | Cohen's d | Direction |")
    lines.append("|---------|-----------|-----------|")
    for a in p["top_associations"]:
        lines.append(
            f"| {a['feature']} | {a['cohens_d']:+.3f} | "
            f"{a['direction']} in this archetype |"
        )
    lines.append("")

    # Feature statistics table (select key features)
    key_features = [
        "gdp_per_capita_pps", "hrst_per_1000", "rd_expenditure_pct_gdp",
        "business_rd_pct_gdp", "epo_patents_per_mio_pop", "employment_rate",
        "broadband_penetration_pct", "enterprise_internet_use",
    ]
    lines.append("### Selected feature statistics (mean +/- SD)\n")
    lines.append("| Feature | Dimension | Mean | SD | EU27 mean | vs EU27 |")
    lines.append("|---------|-----------|------|----|-----------|---------|")
    for fc in key_features:
        if fc not in p["feature_stats"]:
            continue
        s = p["feature_stats"][fc]
        pct = f"{s['pct_vs_eu27']:+.1f}%" if s["pct_vs_eu27"] is not None else "N/A"
        lines.append(
            f"| {s['label']} | {s['dimension']} | "
            f"{_fmt(s['mean'])} | {_fmt(s['sd'])} | "
            f"{_fmt(s['eu27_mean'])} | {pct} |"
        )
    lines.append("")

    # Typical regions
    lines.append("### Most representative regions (smallest centroid distance)\n")
    lines.append("| NUTS2 code | Region | Country | Distance | Silhouette |")
    lines.append("|------------|--------|---------|----------|------------|")
    for r in p["typical_regions"]:
        cc   = r["country_code"]
        name = r["nuts2_name"]
        lines.append(
            f"| {r['nuts2_code']} | {name} | {COUNTRY_NAMES.get(cc, cc)} | "
            f"{r['distance_to_centroid']:.4f} | {r['silhouette_sample']:.4f} |"
        )
    lines.append("")

    # Within-archetype heterogeneity
    if p["high_cv_features"]:
        lines.append("### Within-archetype heterogeneity\n")
        lines.append(
            "The following features show high within-archetype variation "
            "(coefficient of variation > 0.50), indicating that the archetype "
            "label masks substantial internal diversity:\n"
        )
        for h in p["high_cv_features"]:
            lines.append(f"- **{h['label']}**: CV = {h['cv']:.3f}")
        lines.append("")

    # Data quality
    lines.append("### Data quality notes\n")
    lines.append(
        f"- Mean data quality score: {p['mean_data_quality']:.3f} (scale 0-1)\n"
        f"- Mean silhouette coefficient: {p['mean_silhouette']:.4f}\n"
        f"- Regions with dimensionality warning (>3 imputed analytical features): "
        f"{p['n_dimensionality_warn']}\n"
    )

    return "\n".join(lines)


# ---- Figures -----------------------------------------------------------------

def plot_feature_comparison(df: pd.DataFrame) -> None:
    """Paired bar chart: mean +/- SD per feature, A0 vs A1 vs EU27."""
    feat_order = [f for f in FEATURE_COLS if f != "population_density"]
    labels     = [FEATURE_LABELS[f] for f in feat_order]

    means, sds = {}, {}
    for aid in [0, 1]:
        sub = df[df["archetype_id"] == aid]
        means[aid] = [sub[f].mean() for f in feat_order]
        sds[aid]   = [sub[f].std(ddof=1) for f in feat_order]
    eu27_means = [df[f].mean() for f in feat_order]

    # Normalize to EU27 mean for comparability (index: EU27 = 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = {
            aid: [
                m / e if e != 0 else np.nan
                for m, e in zip(means[aid], eu27_means)
            ]
            for aid in [0, 1]
        }
        idx_sd = {
            aid: [
                s / e if e != 0 else np.nan
                for s, e in zip(sds[aid], eu27_means)
            ]
            for aid in [0, 1]
        }

    n = len(feat_order)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(18, 6))
    for i, (aid, offset, lbl) in enumerate([(0, -w / 2, "A1: Catching-up"),
                                              (1,  w / 2, "A2: High-performance")]):
        vals = np.array(idx[aid], dtype=float)
        errs = np.array(idx_sd[aid], dtype=float)
        bar_colors = [DIM_COLORS.get(DIMENSION_MAP.get(f, "Context"), "#aaa")
                      for f in feat_order]
        ax.bar(x + offset, vals, w, yerr=errs, capsize=2,
               color=bar_colors, alpha=(0.55 if aid == 0 else 0.9),
               label=lbl, error_kw=dict(linewidth=0.8))

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="EU27 mean (index = 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Feature value (index: EU27 mean = 1.0)", fontsize=9)
    ax.set_title(
        "Mean feature values by archetype (indexed to EU27 mean)\n"
        "Error bars: +/- 1 SD  |  EU NUTS2 Regional Innovation Panel  |  Source: Eurostat",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper right")

    # Dimension colour legend
    dim_patches = [mpatches.Patch(color=DIM_COLORS[d], label=d)
                   for d in ["Prosperity", "Talent", "Infra", "Innovation"]]
    ax.legend(handles=dim_patches + [
        mpatches.Patch(color="white", label=""),
        mpatches.Patch(color="grey", alpha=0.55, label="A1: Catching-up"),
        mpatches.Patch(color="grey", alpha=0.9,  label="A2: High-performance"),
    ], fontsize=8, ncol=3, loc="upper right")

    plt.tight_layout()
    path = FIG_DIR / "p8_feature_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_score_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: regions (sorted by cluster score) x 4 dimension scores."""
    df_s = df.sort_values(["archetype_id", "score_cluster"], ascending=[True, False])
    mat  = df_s[MM_COLS].values

    # Divider between archetypes
    div_idx = int((df_s["archetype_id"] == 0).sum())

    fig, ax = plt.subplots(figsize=(8, 14))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.axhline(div_idx - 0.5, color="white", linewidth=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Prosperity", "Talent", "Infra.", "Innovation"], fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel("NUTS2 regions (sorted by archetype then innovation score)", fontsize=8)

    # Archetype labels on y-axis
    ax.text(-0.5, div_idx / 2, "A1: Catching-up",
            va="center", ha="right", fontsize=8, color=ARCHETYPE_COLORS[0],
            transform=ax.transData, rotation=90)
    ax.text(-0.5, div_idx + (len(df) - div_idx) / 2, "A2: High-performance",
            va="center", ha="right", fontsize=8, color=ARCHETYPE_COLORS[1],
            transform=ax.transData, rotation=90)

    plt.colorbar(im, ax=ax, shrink=0.4, label="Normalised score [0, 1]",
                 orientation="horizontal", pad=0.02)
    ax.set_title(
        "Normalised dimension scores per NUTS2 region\n"
        "EU Regional Innovation Panel  |  Source: Eurostat",
        fontsize=10,
    )
    plt.tight_layout()
    path = FIG_DIR / "p8_score_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


# ---- Main -------------------------------------------------------------------

def main() -> None:
    with pipeline_step("p8_load"):
        df  = pd.read_parquet(GOLD_PATH)
        eff = pd.read_csv(P6_EFF) if P6_EFF.exists() else None
        log.info("data loaded", shape=df.shape, eff_available=(eff is not None))
        eu27_gdp_mean = float(df["gdp_per_capita_pps"].mean())

    with pipeline_step("p8_build_profiles"):
        profiles = [
            build_archetype_profile(df, aid, eff)
            for aid in sorted(df["archetype_id"].unique())
        ]
        log.info("profiles built", n_archetypes=len(profiles))

    with pipeline_step("p8_write_markdown"):
        md_path = ANA_DIR / "p8_archetype_profiles.md"
        header  = (
            "# EU NUTS2 Regional Innovation Panel -- Archetype Profiles\n\n"
            "> **Non-causal framing**: This document presents observational, "
            "cross-sectional, descriptive findings. All associations are "
            "characterised as correlates or patterns -- no causal claims are made.\n\n"
            f"> **Data source**: Eurostat (NUTS 2021 vintage). "
            f"N = {len(df)} NUTS2 regions across 27 EU member states.\n\n"
        )
        body = "\n\n---\n\n".join(
            profile_to_markdown(p, eu27_gdp_mean) for p in profiles
        )
        md_path.write_text(header + body, encoding="utf-8")
        log.info("markdown written", path=str(md_path), bytes=md_path.stat().st_size)

    with pipeline_step("p8_write_json"):
        json_path = ANA_DIR / "p8_archetype_profiles.json"
        def _json_safe(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        json_path.write_text(
            json.dumps(profiles, ensure_ascii=False, indent=2, default=_json_safe),
            encoding="utf-8",
        )
        log.info("JSON written", path=str(json_path))

    with pipeline_step("p8_region_table"):
        tbl = df[[
            "nuts2_name", "country_code",
            "archetype_id", "archetype_label",
            "score_cost", "score_talent", "score_infra", "score_cluster",
            "score_cost_minmax", "score_talent_minmax",
            "score_infra_minmax", "score_cluster_minmax",
            "silhouette_sample", "distance_to_centroid",
            "is_transition_region", "is_insufficient_data",
            "dimensionality_warning", "data_quality_score",
            "gdp_per_capita_pps", "hrst_per_1000", "rd_expenditure_pct_gdp",
            "epo_patents_per_mio_pop", "employment_rate",
            "broadband_penetration_pct",
        ]].copy()
        # Regions with silhouette < 0.20 are near the cluster boundary;
        # their archetype assignment carries more uncertainty.
        tbl["uncertain_assignment"] = tbl["silhouette_sample"] < 0.20
        tbl.index.name = "nuts2_code"
        tbl = tbl.sort_values(["country_code", "archetype_id"])
        csv_path = ANA_DIR / "p8_region_table.csv"
        tbl.to_csv(csv_path, encoding="utf-8")
        n_uncertain = int(tbl["uncertain_assignment"].sum())
        log.info("region table written",
                 path=str(csv_path), n_rows=len(tbl), n_cols=len(tbl.columns),
                 n_uncertain_assignment=n_uncertain)

    with pipeline_step("p8_figures"):
        plot_feature_comparison(df)
        plot_score_heatmap(df)

    # ---- Gate check ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("P8 GATE CHECK")
    print("=" * 65)

    md_ok   = (ANA_DIR / "p8_archetype_profiles.md").exists()
    json_ok = (ANA_DIR / "p8_archetype_profiles.json").exists()
    csv_ok  = (ANA_DIR / "p8_region_table.csv").exists()
    fig1_ok = (FIG_DIR / "p8_feature_comparison.png").exists()
    fig2_ok = (FIG_DIR / "p8_score_heatmap.png").exists()

    checks = [
        ("p8_archetype_profiles.md",   md_ok),
        ("p8_archetype_profiles.json", json_ok),
        ("p8_region_table.csv",        csv_ok),
        ("p8_feature_comparison.png",  fig1_ok),
        ("p8_score_heatmap.png",       fig2_ok),
    ]
    all_pass = all(ok for _, ok in checks)
    for name, ok in checks:
        print(f"  {'[PASS]' if ok else '[FAIL]'}  {name}")

    print(f"\n  Archetypes profiled   : {len(profiles)}")
    for p in profiles:
        print(f"  A{p['archetype_id']+1}: {p['archetype_label']}")
        print(f"       n={p['n_regions']}, {p['n_countries']} countries, "
              f"{p['n_transition_regions']} transition regions, "
              f"mean sil={p['mean_silhouette']:.4f}")
        print(f"       Top countries: {', '.join(p['top5_countries'][:3])}")

    print(f"\n  Region table rows     : {len(tbl)}")
    print(f"\nGATE_P8={'PASS' if all_pass else 'FAIL'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
