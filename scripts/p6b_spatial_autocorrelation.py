"""
Phase 6b -- Spatial Autocorrelation Analysis
Tests whether archetype membership and dimension scores are spatially clustered
across EU NUTS2 regions using Queen contiguity weights derived from the Eurostat
GeoJSON. All findings are descriptive and non-causal.

Methods:
  - Queen contiguity spatial weights (libpysal)
  - Global Moran's I with 999 permutations (esda.Moran)
  - Local Moran's I (LISA) with 999 permutations (esda.Moran_Local)

Outputs:
  analysis/p6b_global_moran.csv  -- Global Moran's I for archetype_id + 4 scores
  analysis/p6b_lisa_results.csv  -- Per-region LISA statistics and cluster category
  figures/p6b_moran_scatter.png  -- Moran scatter plot (archetype_id)
  figures/p6b_lisa_map.png       -- LISA cluster map (HH/LL/LH/HL/NS)
  figures/p6b_score_moran.png    -- Global Moran's I bar chart for all tested variables
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import libpysal.weights as lps_w
import esda
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

N_PERMUTATIONS = 999
SIGNIFICANCE   = 0.05

SCORE_VARS = {
    "archetype_id":       "Archetype (0/1)",
    "score_cost":         "Prosperity score",
    "score_talent":       "Talent score",
    "score_infra":        "Digital infra. score",
    "score_cluster":      "Innovation cluster score",
}

LISA_COLORS = {
    "HH": "#d7191c",
    "LL": "#2c7bb6",
    "LH": "#abd9e9",
    "HL": "#fdae61",
    "NS": "#eeeeee",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def lisa_category(I_local: float, z: float, p: float,
                  sig: float = SIGNIFICANCE) -> str:
    if p > sig:
        return "NS"
    if z > 0 and I_local > 0:
        return "HH"
    if z < 0 and I_local > 0:
        return "LL"
    if z < 0 and I_local < 0:
        return "LH"
    return "HL"


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_moran_scatter(
    gdf: gpd.GeoDataFrame,
    w: lps_w.W,
    var: str,
    label: str,
    moran_result: esda.Moran,
) -> None:
    y  = gdf[var].values
    wy = lps_w.lag_spatial(w, y)
    y_std  = (y  - y.mean())  / y.std()
    wy_std = (wy - wy.mean()) / wy.std()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_std, wy_std, s=14, alpha=0.55, color="#4c72b0", linewidths=0)

    # Quadrant lines
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)

    # Regression line (slope = Moran's I)
    m = np.polyfit(y_std, wy_std, 1)
    x_line = np.linspace(y_std.min(), y_std.max(), 200)
    ax.plot(x_line, np.polyval(m, x_line), color="#dd8452", linewidth=1.8,
            label=f"Slope = Moran's I = {moran_result.I:.4f}")

    ax.set_xlabel(f"Standardised {label}", fontsize=10)
    ax.set_ylabel(f"Spatial lag of {label}", fontsize=10)
    ax.set_title(
        f"Moran scatter — {label}\n"
        f"I = {moran_result.I:.4f},  p = {moran_result.p_sim:.4f} "
        f"(permutations = {N_PERMUTATIONS})",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    path = FIG_DIR / "p6b_moran_scatter.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("saved figure", path=str(path))


def plot_score_moran(global_results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels   = [r["label"] for r in global_results]
    I_vals   = [r["moran_I"] for r in global_results]
    sig_mask = [r["p_sim"] <= SIGNIFICANCE for r in global_results]
    colors   = ["#d7191c" if s else "#aaaaaa" for s in sig_mask]

    bars = ax.barh(labels, I_vals, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Global Moran's I", fontsize=10)
    ax.set_title(
        f"Spatial autocorrelation — Global Moran's I\n"
        f"(Queen contiguity weights, {N_PERMUTATIONS} permutations; "
        f"red = p ≤ {SIGNIFICANCE})",
        fontsize=10,
    )
    ax.tick_params(labelsize=9)

    for bar, r in zip(bars, global_results):
        sign = "+" if r["moran_I"] >= 0 else ""
        lbl  = f" {sign}{r['moran_I']:.3f} (p={r['p_sim']:.3f})"
        ax.text(
            bar.get_width() + 0.002 if r["moran_I"] >= 0 else bar.get_width() - 0.002,
            bar.get_y() + bar.get_height() / 2,
            lbl, va="center",
            ha="left" if r["moran_I"] >= 0 else "right",
            fontsize=8,
        )
    plt.tight_layout()
    path = FIG_DIR / "p6b_score_moran.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("saved figure", path=str(path))


def plot_lisa_map(gdf: gpd.GeoDataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for cat, color in LISA_COLORS.items():
        subset = gdf[gdf["lisa_category"] == cat]
        if len(subset):
            subset.plot(ax=ax, color=color, linewidth=0.2, edgecolor="white")

    patches = [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in [
            ("HH — High archetype, high neighbours", LISA_COLORS["HH"]),
            ("LL — Low archetype, low neighbours",   LISA_COLORS["LL"]),
            ("LH — Low archetype, high neighbours",  LISA_COLORS["LH"]),
            ("HL — High archetype, low neighbours",  LISA_COLORS["HL"]),
            ("NS — Not significant",                 LISA_COLORS["NS"]),
        ]
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower left",
              framealpha=0.9, edgecolor="grey")
    ax.set_title(
        f"LISA cluster map — Archetype membership\n"
        f"(Queen contiguity, p ≤ {SIGNIFICANCE}, {N_PERMUTATIONS} permutations)",
        fontsize=11,
    )
    ax.set_axis_off()
    plt.tight_layout()
    path = FIG_DIR / "p6b_lisa_map.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("saved figure", path=str(path))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    with pipeline_step("p6b_load"):
        gold = pd.read_parquet(GOLD_PATH)
        gdf_raw = gpd.read_file(GEOJSON_PATH)
        log.info("loaded", gold_shape=gold.shape, geojson_features=len(gdf_raw))

    # ── Merge Gold attributes onto GDF ────────────────────────────────────────
    with pipeline_step("p6b_merge"):
        gold_reset = gold.reset_index()
        gdf = gdf_raw[gdf_raw["NUTS_ID"].isin(gold_reset["nuts2_code"])].copy()
        gdf = gdf.merge(
            gold_reset[["nuts2_code"] + list(SCORE_VARS.keys())],
            left_on="NUTS_ID", right_on="nuts2_code", how="inner",
        )
        gdf = gdf.set_index("nuts2_code").sort_index()
        # Ensure projected CRS (EPSG:3035 — ETRS89 / LAEA) for accurate contiguity
        if gdf.crs is None or gdf.crs.to_epsg() != 3035:
            gdf = gdf.to_crs(epsg=3035)
        log.info("merged GDF", n_regions=len(gdf))

    # ── Build spatial weights ─────────────────────────────────────────────────
    with pipeline_step("p6b_weights"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = lps_w.Queen.from_dataframe(gdf, silence_warnings=True)
        w.transform = "r"   # row-standardise
        n_islands = len(w.islands)
        log.info("Queen weights built",
                 n_obs=w.n, mean_neighbors=round(w.mean_neighbors, 2),
                 n_islands=n_islands)
        if n_islands:
            log.warning("island regions (no neighbours)", codes=w.islands)

    # ── Global Moran's I ──────────────────────────────────────────────────────
    with pipeline_step("p6b_global_moran"):
        global_results = []
        for var, label in SCORE_VARS.items():
            y = gdf[var].values.astype(float)
            mi = esda.Moran(y, w, permutations=N_PERMUTATIONS)
            global_results.append({
                "variable": var,
                "label":    label,
                "moran_I":  round(mi.I, 6),
                "expected_I": round(mi.EI, 6),
                "z_score":  round(mi.z_sim, 4),
                "p_sim":    round(mi.p_sim, 4),
                "significant": mi.p_sim <= SIGNIFICANCE,
            })
            log.info("Global Moran's I",
                     variable=var, I=round(mi.I, 4),
                     p=round(mi.p_sim, 4), sig=mi.p_sim <= SIGNIFICANCE)
            # Moran scatter only for archetype_id
            if var == "archetype_id":
                moran_archetype = mi
                plot_moran_scatter(gdf, w, var, label, mi)

        global_df = pd.DataFrame(global_results)
        global_df.to_csv(ANA_DIR / "p6b_global_moran.csv", index=False)
        log.info("global Moran saved", path=str(ANA_DIR / "p6b_global_moran.csv"))
        plot_score_moran(global_results)

    # ── Local Moran's I (LISA) on archetype_id ────────────────────────────────
    with pipeline_step("p6b_lisa"):
        y = gdf["archetype_id"].values.astype(float)
        lisa = esda.Moran_Local(y, w, permutations=N_PERMUTATIONS, seed=42)

        gdf["lisa_I"]        = lisa.Is
        gdf["lisa_z"]        = lisa.z_sim
        gdf["lisa_p"]        = lisa.p_sim
        gdf["lisa_category"] = [
            lisa_category(I, z, p)
            for I, z, p in zip(lisa.Is, lisa.z_sim, lisa.p_sim)
        ]

        cat_counts = gdf["lisa_category"].value_counts().to_dict()
        log.info("LISA categories", **cat_counts)

        lisa_df = gdf[["lisa_I", "lisa_z", "lisa_p", "lisa_category",
                        "archetype_id"]].reset_index()
        lisa_df.to_csv(ANA_DIR / "p6b_lisa_results.csv", index=False)
        log.info("LISA results saved", path=str(ANA_DIR / "p6b_lisa_results.csv"))
        plot_lisa_map(gdf)

    # ── Gate check ────────────────────────────────────────────────────────────
    n_sig_global = sum(r["significant"] for r in global_results)
    archetype_I  = next(r for r in global_results if r["variable"] == "archetype_id")

    print("\n" + "=" * 65)
    print("P6b SPATIAL AUTOCORRELATION — GATE CHECK")
    print("=" * 65)
    print(f"\n  Spatial weights : Queen contiguity, n={w.n}, "
          f"mean neighbours={w.mean_neighbors:.2f}")
    print(f"  Islands (no neighbour): {len(w.islands)}")
    print(f"\n  Global Moran's I results ({N_PERMUTATIONS} permutations):\n")
    print(f"  {'Variable':<30} {'I':>8} {'p':>8} {'Sig':>5}")
    print("  " + "-" * 55)
    for r in global_results:
        sig = "YES" if r["significant"] else "no"
        print(f"  {r['label']:<30} {r['moran_I']:>8.4f} {r['p_sim']:>8.4f} {sig:>5}")

    print(f"\n  Significant (p <= {SIGNIFICANCE}): {n_sig_global}/{len(global_results)}")
    print(f"\n  LISA cluster categories (archetype_id):")
    for cat, n in sorted(cat_counts.items()):
        print(f"    {cat}: {n} regions")

    hh_ll = cat_counts.get("HH", 0) + cat_counts.get("LL", 0)
    print(f"\n  Spatially coherent assignments (HH+LL): {hh_ll} regions")

    archetype_pass = archetype_I["significant"]
    print(f"\n  Archetype Moran's I significant: {'PASS' if archetype_pass else 'FAIL'}")
    print(f"\n  Artifacts:")
    print(f"    analysis/p6b_global_moran.csv")
    print(f"    analysis/p6b_lisa_results.csv")
    print(f"    figures/p6b_moran_scatter.png")
    print(f"    figures/p6b_lisa_map.png")
    print(f"    figures/p6b_score_moran.png")
    status = "PASS" if archetype_pass else "WARN"
    print(f"\nGATE_P6b={status}")
    print("=" * 65)


if __name__ == "__main__":
    main()
