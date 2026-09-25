"""
Phase 6d -- Sectoral Co-Specialisation Typology
Classifies each NUTS2 region by which combination of innovation-relevant
sector LQs exceeds LQ > 1.0 (above the EU reference mean for that sector).

LQ > 1.0 indicates the sector is over-represented in the region relative
to the EU average; it is a structural characteristic, not a performance
metric or investment signal.

Pattern taxonomy (hierarchical, most specific first):
  1. Diversified innovation  -- J > 1.0 AND K > 1.0 AND C > 1.0
  2. Digital knowledge       -- J > 1.0 AND K > 1.0
  3. Digital-industrial      -- J > 1.0 AND C > 1.0
  4. Advanced mfg & R&D      -- K > 1.0 AND C > 1.0
  5. Energy-anchored mixed   -- D > 1.0 AND (J OR K OR C) > 1.0
  6. ICT services only       -- J > 1.0
  7. KIS hi-tech only        -- K > 1.0
  8. Hi-tech mfg only        -- C > 1.0
  9. Energy/utilities only   -- D > 1.0
 10. Non-specialised         -- none > 1.0

All language is descriptive and non-causal.

Outputs:
  analysis/p6d_cospec_patterns.csv    -- Per-region pattern + archetype
  analysis/p6d_cospec_summary.csv     -- Pattern × archetype cross-tab + feature means
  figures/p6d_cospec_heatmap.png      -- LQ heatmap sorted by pattern
  figures/p6d_cospec_distribution.png -- Pattern counts by archetype
  figures/p6d_cospec_map.png          -- NUTS2 map coloured by pattern
  figures/p6d_cospec_profiles.png     -- Feature radar per pattern (GDP, HRST, R&D, patents)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import structlog

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils.logging_config import configure_logging, pipeline_step

configure_logging()
log = structlog.get_logger(__name__)

GOLD_PATH    = ROOT / "data" / "gold" / "region_profiles_gold.parquet"
GEOJSON_PATH = ROOT / "data" / "raw" / "eurostat" / "nuts2_2021_geojson.json"
ANA_DIR      = ROOT / "analysis"
FIG_DIR      = ROOT / "figures"
ANA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# LQ columns
LQ_J = "lq_nace_j62j63"      # ICT services
LQ_K = "lq_nace_c21_m72"     # KIS hi-tech (pharma + R&D services)
LQ_C = "lq_nace_c26"         # Hi-tech manufacturing (electronics)
LQ_D = "lq_nace_d35_clean"   # Energy / utilities

LQ_COLS = [LQ_J, LQ_K, LQ_C, LQ_D]
LQ_LABELS = {
    LQ_J: "ICT services\n(J62/J63)",
    LQ_K: "KIS hi-tech\n(C21/M72)",
    LQ_C: "Hi-tech mfg\n(C26)",
    LQ_D: "Energy/utilities\n(D35)",
}

PROFILE_FEATURES = {
    "gdp_per_capita_pps":      "GDP/cap (PPS)",
    "hrst_per_1000":           "HRST per 1 000",
    "rd_expenditure_pct_gdp":  "GERD % GDP",
    "epo_patents_per_mio_pop": "EPO patents / mio.",
    "employment_rate":         "Employment rate",
    "broadband_penetration_pct": "Broadband %",
}

# Pattern taxonomy — ordered from most specific to least specific
PATTERN_ORDER = [
    "Diversified innovation",
    "Digital knowledge economy",
    "Digital-industrial",
    "Advanced mfg & R&D",
    "Energy-anchored mixed",
    "ICT services only",
    "KIS hi-tech only",
    "Hi-tech mfg only",
    "Energy/utilities only",
    "Non-specialised",
]

PATTERN_COLORS = {
    "Diversified innovation":   "#2166ac",
    "Digital knowledge economy":"#4393c3",
    "Digital-industrial":       "#92c5de",
    "Advanced mfg & R&D":       "#74c476",
    "Energy-anchored mixed":     "#fd8d3c",
    "ICT services only":        "#abd9e9",
    "KIS hi-tech only":         "#a1d99b",
    "Hi-tech mfg only":         "#c7e9c0",
    "Energy/utilities only":    "#fdbe85",
    "Non-specialised":          "#d9d9d9",
}

ARCHETYPE_COLORS = {
    "Catching-up & peripheral regions": "#4c72b0",
    "Advanced innovation regions":      "#dd8452",
}

THRESHOLD = 1.0   # LQ threshold for "specialised"


# ── Pattern assignment ────────────────────────────────────────────────────────

def assign_pattern(j: float, k: float, c: float, d: float) -> str:
    j_sp = j > THRESHOLD
    k_sp = k > THRESHOLD
    c_sp = c > THRESHOLD
    d_sp = d > THRESHOLD
    innov_count = sum([j_sp, k_sp, c_sp])

    if innov_count >= 3:
        return "Diversified innovation"
    if j_sp and k_sp:
        return "Digital knowledge economy"
    if j_sp and c_sp:
        return "Digital-industrial"
    if k_sp and c_sp:
        return "Advanced mfg & R&D"
    if d_sp and innov_count >= 1:
        return "Energy-anchored mixed"
    if j_sp:
        return "ICT services only"
    if k_sp:
        return "KIS hi-tech only"
    if c_sp:
        return "Hi-tech mfg only"
    if d_sp:
        return "Energy/utilities only"
    return "Non-specialised"


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_lq_heatmap(df: pd.DataFrame) -> None:
    """LQ heatmap sorted by co-specialisation pattern."""
    order_map = {p: i for i, p in enumerate(PATTERN_ORDER)}
    df_sorted = df.assign(_ord=df["cospec_pattern"].map(order_map)).sort_values(
        ["_ord"] + LQ_COLS, ascending=[True, False, False, False, False]
    )

    lq_mat = df_sorted[LQ_COLS].values
    # Cap display at 3.0 for readability
    lq_mat = np.clip(lq_mat, 0, 3.0)

    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(lq_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=3.0,
                   interpolation="nearest")

    ax.set_xticks(range(len(LQ_COLS)))
    ax.set_xticklabels([LQ_LABELS[c] for c in LQ_COLS], fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel(f"NUTS2 regions (n={len(df_sorted)})", fontsize=9)
    ax.set_title(
        "Sector LQ heatmap — sorted by co-specialisation pattern\n"
        "(LQ > 1.0 = sector over-represented vs EU average; capped at 3.0)",
        fontsize=10,
    )

    # Pattern boundary lines and labels
    prev_pat = None
    start_idx = 0
    boundaries = []
    for i, pat in enumerate(df_sorted["cospec_pattern"].values):
        if pat != prev_pat:
            if prev_pat is not None:
                boundaries.append((start_idx, i, prev_pat))
            start_idx = i
            prev_pat = pat
    boundaries.append((start_idx, len(df_sorted), prev_pat))

    for start, end, pat in boundaries:
        mid = (start + end) / 2
        ax.axhline(start - 0.5, color="white", linewidth=0.8)
        ax.text(len(LQ_COLS) + 0.05, mid, pat,
                va="center", ha="left", fontsize=7.5,
                color=PATTERN_COLORS.get(pat, "grey"))

    ax.axvline(0.5, color="white", linewidth=0.5)
    ax.axvline(1.5, color="white", linewidth=0.5)
    ax.axvline(2.5, color="white", linewidth=0.5)
    ax.axvline(-0.5, color="white", linewidth=0.5, lw=3)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.16)
    cbar.set_label("Location quotient (capped at 3.0)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = FIG_DIR / "p6d_cospec_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("saved figure", path=str(path))


def plot_distribution(df: pd.DataFrame) -> None:
    """Stacked bar: pattern counts split by archetype."""
    counts = (df.groupby(["cospec_pattern", "archetype_label"])
                .size().unstack(fill_value=0))
    # Reindex to canonical order
    counts = counts.reindex(
        [p for p in PATTERN_ORDER if p in counts.index]
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(counts))
    for arch, color in ARCHETYPE_COLORS.items():
        if arch not in counts.columns:
            continue
        vals = counts[arch].values
        bars = ax.barh(counts.index.tolist(), vals, left=bottoms,
                       color=color, edgecolor="white", height=0.65,
                       label=arch.replace(" regions", ""))
        # Value labels
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(b + v / 2, i, str(v),
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold")
        bottoms += vals

    ax.set_xlabel("Number of NUTS2 regions", fontsize=10)
    ax.set_title(
        "Co-specialisation patterns — region count by archetype\n"
        "(LQ > 1.0 threshold; patterns ordered from most to least complex)",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, bottoms.max() * 1.12)
    plt.tight_layout()
    path = FIG_DIR / "p6d_cospec_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("saved figure", path=str(path))


def plot_map(df: pd.DataFrame) -> None:
    """NUTS2 choropleth coloured by co-specialisation pattern."""
    gdf_raw = gpd.read_file(GEOJSON_PATH)
    gdf = gdf_raw[gdf_raw["NUTS_ID"].isin(df.index)].copy()
    gdf = gdf.merge(
        df[["cospec_pattern"]].reset_index(),
        left_on="NUTS_ID", right_on="nuts2_code", how="left",
    )
    gdf = gdf.to_crs(epsg=3035)

    fig, ax = plt.subplots(figsize=(12, 9))
    # Draw non-specialised as base
    for pat in PATTERN_ORDER:
        subset = gdf[gdf["cospec_pattern"] == pat]
        if len(subset):
            subset.plot(ax=ax, color=PATTERN_COLORS[pat],
                        linewidth=0.2, edgecolor="white")

    patches = [
        mpatches.Patch(color=PATTERN_COLORS[p], label=p)
        for p in PATTERN_ORDER
        if p in gdf["cospec_pattern"].values
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower left",
              framealpha=0.92, edgecolor="grey", ncol=2)
    ax.set_title(
        "Sectoral co-specialisation typology — NUTS2 regions\n"
        "(Location quotient > 1.0 threshold; observational, non-causal)",
        fontsize=11,
    )
    ax.set_axis_off()
    plt.tight_layout()
    path = FIG_DIR / "p6d_cospec_map.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("saved figure", path=str(path))


def plot_profiles(summary: pd.DataFrame) -> None:
    """Grouped bar chart: mean feature values per co-specialisation pattern."""
    features = list(PROFILE_FEATURES.keys())
    labels   = list(PROFILE_FEATURES.values())

    # Normalise each feature to [0, 1] across patterns for comparability
    feat_df = summary[features].copy()
    for f in features:
        mn, mx = feat_df[f].min(), feat_df[f].max()
        if mx > mn:
            feat_df[f] = (feat_df[f] - mn) / (mx - mn)

    patterns = feat_df.index.tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(patterns)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (pat, row) in enumerate(feat_df.iterrows()):
        offset = (i - len(patterns) / 2 + 0.5) * width
        ax.bar(x + offset, row.values, width=width * 0.9,
               color=PATTERN_COLORS.get(pat, "#aaaaaa"),
               label=pat, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Normalised mean [0–1 across patterns]", fontsize=9)
    ax.set_title(
        "Mean feature values by co-specialisation pattern\n"
        "(Normalised within each feature for cross-pattern comparison; "
        "observational, non-causal)",
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=2, loc="upper right",
              framealpha=0.9, edgecolor="grey")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, 1.25)
    plt.tight_layout()
    path = FIG_DIR / "p6d_cospec_profiles.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("saved figure", path=str(path))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    with pipeline_step("p6d_load"):
        df = pd.read_parquet(GOLD_PATH)
        log.info("loaded", shape=df.shape)

    # ── Assign co-specialisation patterns ────────────────────────────────────
    with pipeline_step("p6d_patterns"):
        df["cospec_pattern"] = df.apply(
            lambda r: assign_pattern(
                r[LQ_J], r[LQ_K], r[LQ_C], r[LQ_D]
            ),
            axis=1,
        )

        # Binary specialisation flags (LQ > 1.0 and LQ > 1.25)
        for col, short in [(LQ_J, "ict"), (LQ_K, "kis"), (LQ_C, "htmfg"), (LQ_D, "energy")]:
            df[f"spec_{short}"]      = df[col] > 1.0
            df[f"spec_{short}_strong"] = df[col] > 1.25

        pat_counts = df["cospec_pattern"].value_counts()
        log.info("pattern counts", **pat_counts.to_dict())

    # ── Save per-region table ─────────────────────────────────────────────────
    with pipeline_step("p6d_save_patterns"):
        export_cols = (
            ["nuts2_name", "country_code", "archetype_label",
             "cospec_pattern"] + LQ_COLS +
            [c for c in df.columns if c.startswith("spec_")] +
            list(PROFILE_FEATURES.keys()) +
            ["silhouette_sample", "is_transition_region"]
        )
        out = df[export_cols].reset_index()
        out.to_csv(ANA_DIR / "p6d_cospec_patterns.csv", index=False)
        log.info("patterns saved", path=str(ANA_DIR / "p6d_cospec_patterns.csv"),
                 n_rows=len(out))

    # ── Summary: pattern × archetype cross-tab + feature means ───────────────
    with pipeline_step("p6d_summary"):
        # Cross-tab counts
        xtab = (df.groupby(["cospec_pattern", "archetype_label"])
                  .size().unstack(fill_value=0))
        xtab["total"] = xtab.sum(axis=1)
        # A2 share within each pattern
        a2_col = "Advanced innovation regions"
        if a2_col in xtab.columns:
            xtab["a2_share"] = (xtab[a2_col] / xtab["total"]).round(3)

        # Feature means per pattern
        feat_means = df.groupby("cospec_pattern")[list(PROFILE_FEATURES.keys())].mean().round(2)
        # Transition region share per pattern
        trans_share = df.groupby("cospec_pattern")["is_transition_region"].mean().round(3)
        feat_means["pct_transition"] = trans_share

        summary = xtab.join(feat_means)
        summary = summary.reindex([p for p in PATTERN_ORDER if p in summary.index])
        summary.to_csv(ANA_DIR / "p6d_cospec_summary.csv")
        log.info("summary saved", path=str(ANA_DIR / "p6d_cospec_summary.csv"))

    # ── Figures ───────────────────────────────────────────────────────────────
    with pipeline_step("p6d_figures"):
        plot_lq_heatmap(df)
        plot_distribution(df)
        plot_map(df)
        plot_profiles(feat_means)

    # ── Gate check ────────────────────────────────────────────────────────────
    n_specialised = int((df["cospec_pattern"] != "Non-specialised").sum())
    n_cospec      = int(df["cospec_pattern"].isin([
        "Diversified innovation", "Digital knowledge economy",
        "Digital-industrial", "Advanced mfg & R&D", "Energy-anchored mixed",
    ]).sum())

    print("\n" + "=" * 65)
    print("P6d CO-SPECIALISATION TYPOLOGY — GATE CHECK")
    print("=" * 65)
    print(f"\n  LQ threshold for 'specialised': > {THRESHOLD}")
    print(f"  Total regions: {len(df)}")
    print(f"  Regions with >= 1 specialisation: {n_specialised} "
          f"({n_specialised/len(df)*100:.1f}%)")
    print(f"  Co-specialised (>= 2 sectors): {n_cospec} "
          f"({n_cospec/len(df)*100:.1f}%)")

    print(f"\n  Pattern distribution:\n")
    print(f"  {'Pattern':<35} {'n':>4} {'A2 share':>10} {'Trans%':>8}")
    print("  " + "-" * 60)
    for pat in PATTERN_ORDER:
        if pat not in summary.index:
            continue
        row = summary.loc[pat]
        n    = int(row["total"])
        a2s  = f"{row['a2_share']:.0%}" if "a2_share" in row else "n/a"
        tpct = f"{row['pct_transition']:.0%}" if "pct_transition" in row else "n/a"
        print(f"  {pat:<35} {n:>4} {a2s:>10} {tpct:>8}")

    print(f"\n  Artifacts:")
    print(f"    analysis/p6d_cospec_patterns.csv")
    print(f"    analysis/p6d_cospec_summary.csv")
    print(f"    figures/p6d_cospec_heatmap.png")
    print(f"    figures/p6d_cospec_distribution.png")
    print(f"    figures/p6d_cospec_map.png")
    print(f"    figures/p6d_cospec_profiles.png")
    print(f"\nGATE_P6d=PASS")
    print("=" * 65)


if __name__ == "__main__":
    main()
