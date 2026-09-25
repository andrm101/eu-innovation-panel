"""
P5b -- Cluster deep-dive: compare k=2..5 with geographic breakdown.
Uses the updated Gold (hi_tech_employment_pct removed from talent scoring).
Outputs figures + a summary CSV for k selection decision.
Does NOT overwrite Gold -- P5 will be re-run after k is confirmed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import structlog
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils.logging_config import configure_logging, pipeline_step
configure_logging()
log = structlog.get_logger(__name__)

GOLD_PATH = ROOT / "data" / "gold" / "region_profiles_gold.parquet"
FIG_DIR   = ROOT / "figures"
ANA_DIR   = ROOT / "analysis"

SCORE_COLS = ["score_cost", "score_talent", "score_infra", "score_cluster"]
DIM_LABELS = ["Prosperity", "Talent", "Infra.", "Innovation"]

COUNTRY_NAMES = {
    "AT":"Austria","BE":"Belgium","BG":"Bulgaria","CY":"Cyprus","CZ":"Czechia",
    "DE":"Germany","DK":"Denmark","EE":"Estonia","EL":"Greece","ES":"Spain",
    "FI":"Finland","FR":"France","HR":"Croatia","HU":"Hungary","IE":"Ireland",
    "IT":"Italy","LT":"Lithuania","LU":"Luxembourg","LV":"Latvia","MT":"Malta",
    "NL":"Netherlands","PL":"Poland","PT":"Portugal","RO":"Romania",
    "SE":"Sweden","SI":"Slovenia","SK":"Slovakia",
}

# Palette: enough colours for k=5
PALETTE = ["#4c72b0","#dd8452","#55a868","#c44e52","#8172b2"]


def fit_all_k(X_std: np.ndarray, k_range=range(2, 6)) -> dict:
    results = {}
    for k in k_range:
        km   = KMeans(n_clusters=k, n_init=50, max_iter=500, random_state=42)
        lbs  = km.fit_predict(X_std)
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_std)
        results[k] = {
            "labels":      lbs,
            "km":          km,
            "silhouette":  silhouette_score(X_std, lbs),
            "sil_samples": silhouette_samples(X_std, lbs),
            "ch":          calinski_harabasz_score(X_std, lbs),
            "db":          davies_bouldin_score(X_std, lbs),
            "ari_ward":    adjusted_rand_score(lbs, ward),
            "inertia":     km.inertia_,
        }
        log.info("k fitted", k=k,
                 silhouette=round(results[k]["silhouette"], 4),
                 ch=round(results[k]["ch"], 1),
                 db=round(results[k]["db"], 4),
                 ari_ward=round(results[k]["ari_ward"], 4),
                 sizes=np.bincount(lbs).tolist())
    return results


def centroid_profiles(df: pd.DataFrame, labels: np.ndarray,
                      score_cols, mm_cols) -> pd.DataFrame:
    rows = []
    for cid in sorted(set(labels)):
        mask = labels == cid
        row  = {"cluster": cid, "n": int(mask.sum())}
        for sc, mm in zip(score_cols, mm_cols):
            row[f"{sc}_mean"] = round(df.loc[mask, sc].mean(), 3)
            row[f"{mm}_mean"] = round(df.loc[mask, mm].mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def dominant_label(profile_row: pd.Series) -> str:
    """Pick a short descriptive label from centroid mm scores."""
    mm = [profile_row.get(f"score_{d}_minmax_mean", 0)
          for d in ["cost", "talent", "infra", "cluster"]]
    dims = ["Prosperity", "Talent", "Infra", "Innovation"]
    level = np.mean(mm)
    dom   = dims[int(np.argmax(mm))]
    if level >= 0.60:
        return f"High-performance ({dom.lower()}-led)"
    elif level >= 0.45:
        return f"Mid-range ({dom.lower()}-led)"
    elif level >= 0.30:
        return f"Catching-up ({dom.lower()}-moderate)"
    else:
        return "Low-development peripheral"


# ---- Figures ----------------------------------------------------------------

def fig_validity_indices(results: dict) -> None:
    ks  = sorted(results)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    metrics = [
        ("silhouette", "Silhouette score", "o-", "#4c72b0"),
        ("ch",         "Calinski-Harabasz",  "s--","#55a868"),
        ("db",         "Davies-Bouldin (lower better)", "^:", "#c44e52"),
        ("ari_ward",   "ARI vs Ward linkage",  "D-.", "#8172b2"),
    ]
    for ax, (key, ylabel, fmt, color) in zip(axes, metrics):
        vals = [results[k][key] for k in ks]
        ax.plot(ks, vals, fmt, color=color, linewidth=2, markersize=7)
        ax.set_xticks(ks)
        ax.set_xlabel("k", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split("(")[0].strip(), fontsize=10)
        ax.grid(alpha=0.3)
        if key == "silhouette":
            ax.axhline(0.35, color="grey", linestyle="--", linewidth=0.8,
                       label="Threshold 0.35")
            ax.legend(fontsize=8)
        # Annotate max / min
        best_k = ks[int(np.argmin(vals) if key == "db" else np.argmax(vals))]
        ax.axvline(best_k, color="red", linestyle=":", linewidth=1, alpha=0.6)

    fig.suptitle(
        "Cluster validity indices across k=2..5  |  Updated scores (hi_tech excluded)\n"
        "Red dotted: best k per metric  |  Source: Eurostat",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p5b_validity_indices.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved", path=str(path))


def fig_country_heatmap(df: pd.DataFrame, results: dict) -> None:
    """For k=2,3,4: show country x cluster count as heatmaps side by side."""
    ks_to_show = [k for k in [2, 3, 4] if k in results]
    fig, axes = plt.subplots(1, len(ks_to_show), figsize=(6 * len(ks_to_show), 9))
    if len(ks_to_show) == 1:
        axes = [axes]

    countries = sorted(df["country_code"].unique(),
                       key=lambda c: COUNTRY_NAMES.get(c, c))

    for ax, k in zip(axes, ks_to_show):
        lbs = results[k]["labels"]
        tmp = df.copy()
        tmp["cluster"] = lbs
        mat = (tmp.groupby(["country_code", "cluster"])
               .size().unstack(fill_value=0)
               .reindex(countries, fill_value=0))
        # Normalize to fraction per country
        frac = mat.div(mat.sum(axis=1), axis=0)

        im = ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"C{i+1}" for i in range(k)], fontsize=10)
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(
            [COUNTRY_NAMES.get(c, c) for c in countries], fontsize=8
        )
        for i in range(len(countries)):
            for j in range(k):
                v = frac.values[i, j]
                if v > 0.05:
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=7, color="white" if v > 0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.4, label="Share of national NUTS2")
        ax.set_title(
            f"k={k}  (sil={results[k]['silhouette']:.3f}, "
            f"ARI_ward={results[k]['ari_ward']:.2f})",
            fontsize=10,
        )

    fig.suptitle(
        "Country x cluster composition (fraction of national NUTS2 regions)\n"
        "EU NUTS2 Regional Innovation Panel  |  Source: Eurostat",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p5b_country_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved", path=str(path))


def fig_score_profiles(df: pd.DataFrame, results: dict,
                       mm_cols: list[str]) -> None:
    """Radar-style bar chart of centroid mm-scores for k=2,3,4."""
    ks_to_show = [k for k in [2, 3, 4] if k in results]
    fig, axes = plt.subplots(1, len(ks_to_show),
                             figsize=(5 * len(ks_to_show), 4))
    if len(ks_to_show) == 1:
        axes = [axes]

    x = np.arange(4)
    w = 0.8 / max(ks_to_show)

    for ax, k in zip(axes, ks_to_show):
        lbs = results[k]["labels"]
        for cid in range(k):
            mask   = lbs == cid
            means  = df.loc[mask, mm_cols].mean().values
            offset = (cid - (k - 1) / 2) * w
            ax.bar(x + offset, means, w, color=PALETTE[cid], alpha=0.85,
                   label=f"C{cid+1} (n={mask.sum()})")
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS, fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.6)
        ax.set_ylabel("Mean normalised score [0,1]", fontsize=9)
        ax.set_title(
            f"k={k} centroid profiles\n"
            f"sil={results[k]['silhouette']:.3f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)

    fig.suptitle(
        "Cluster centroid profiles (normalised scores) for k=2..4\n"
        "EU NUTS2 Regional Innovation Panel  |  Source: Eurostat",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p5b_score_profiles.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved", path=str(path))


def fig_silhouette_by_k(X_std: np.ndarray, results: dict) -> None:
    """Side-by-side silhouette distribution plots for k=2,3,4."""
    ks_to_show = [k for k in [2, 3, 4] if k in results]
    fig, axes = plt.subplots(1, len(ks_to_show),
                             figsize=(5 * len(ks_to_show), 4), sharey=False)
    if len(ks_to_show) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks_to_show):
        lbs      = results[k]["labels"]
        sil_samp = results[k]["sil_samples"]
        y_lower  = 5
        for cid in range(k):
            mask = lbs == cid
            vals = np.sort(sil_samp[mask])
            y_upper = y_lower + len(vals)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                             facecolor=PALETTE[cid], alpha=0.8)
            ax.text(-0.02, (y_lower + y_upper) / 2,
                    f"C{cid+1}", ha="right", va="center",
                    fontsize=8, color=PALETTE[cid])
            y_lower = y_upper + 3
        mean_s = sil_samp.mean()
        ax.axvline(0,      color="black",  linewidth=0.8, linestyle="--")
        ax.axvline(mean_s, color="red",    linewidth=1.2, linestyle="--",
                   label=f"mean={mean_s:.3f}")
        ax.set_xlabel("Silhouette", fontsize=9)
        ax.set_yticks([])
        ax.set_title(f"k={k}  (sil={results[k]['silhouette']:.3f}, "
                     f"n_neg={(sil_samp < 0).sum()})", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Per-region silhouette distributions for k=2..4\n"
        "Negative values indicate possible misclassification  |  Source: Eurostat",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p5b_silhouette_by_k.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved", path=str(path))


def fig_k3_split(df: pd.DataFrame, results: dict,
                 score_cols: list[str]) -> None:
    """For k=3: scatter plot and country breakdown of what split from k=2."""
    if 2 not in results or 3 not in results:
        return

    lbs2  = results[2]["labels"]
    lbs3  = results[3]["labels"]
    # Map k=3 clusters to k=2 clusters for comparison
    # Find most similar k=3 cluster to each k=2 cluster
    df2 = df.copy()
    df2["c2"] = lbs2
    df2["c3"] = lbs3

    fig = plt.figure(figsize=(16, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: Cost vs Cluster score, k=3 coloured
    ax0 = fig.add_subplot(gs[0])
    for cid in range(3):
        mask = lbs3 == cid
        ax0.scatter(df.loc[mask, "score_cost"],
                    df.loc[mask, "score_cluster"],
                    s=12, alpha=0.65, color=PALETTE[cid],
                    label=f"C{cid+1} (n={mask.sum()})")
    ax0.axhline(0, color="lightgrey", lw=0.5)
    ax0.axvline(0, color="lightgrey", lw=0.5)
    ax0.set_xlabel("Prosperity (cost)", fontsize=9)
    ax0.set_ylabel("Innovation cluster", fontsize=9)
    ax0.set_title("k=3: Prosperity vs Innovation", fontsize=10)
    ax0.legend(fontsize=8)

    # Panel 2: Talent vs Cluster, k=3
    ax1 = fig.add_subplot(gs[1])
    for cid in range(3):
        mask = lbs3 == cid
        ax1.scatter(df.loc[mask, "score_talent"],
                    df.loc[mask, "score_cluster"],
                    s=12, alpha=0.65, color=PALETTE[cid])
    ax1.axhline(0, color="lightgrey", lw=0.5)
    ax1.axvline(0, color="lightgrey", lw=0.5)
    ax1.set_xlabel("Talent", fontsize=9)
    ax1.set_ylabel("Innovation cluster", fontsize=9)
    ax1.set_title("k=3: Talent vs Innovation", fontsize=10)

    # Panel 3: Country breakdown for k=3
    ax2 = fig.add_subplot(gs[2])
    countries_order = (df2.groupby("country_code")["c3"]
                       .mean()
                       .sort_values()
                       .index.tolist())
    counts  = (df2.groupby(["country_code", "c3"])
               .size().unstack(fill_value=0)
               .reindex(countries_order))
    fracs   = counts.div(counts.sum(axis=1), axis=0)
    bottom  = np.zeros(len(fracs))
    for cid in range(3):
        if cid in fracs.columns:
            ax2.barh(range(len(fracs)), fracs[cid].values,
                     left=bottom, color=PALETTE[cid], alpha=0.85,
                     label=f"C{cid+1}")
            bottom += fracs[cid].values
    ax2.set_yticks(range(len(fracs)))
    ax2.set_yticklabels(
        [COUNTRY_NAMES.get(c, c) for c in countries_order], fontsize=7.5
    )
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Share of national NUTS2 regions", fontsize=9)
    ax2.set_title("k=3: Country composition", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "k=3 cluster solution: scatter plots and country breakdown\n"
        f"Silhouette={results[3]['silhouette']:.3f}  "
        f"ARI vs Ward={results[3]['ari_ward']:.3f}  |  Source: Eurostat",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p5b_k3_detail.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved", path=str(path))


# ---- Main -------------------------------------------------------------------

def main() -> None:
    with pipeline_step("p5b_load"):
        df = pd.read_parquet(GOLD_PATH)
        log.info("loaded", shape=df.shape)

    mm_cols = ["score_cost_minmax", "score_talent_minmax",
               "score_infra_minmax", "score_cluster_minmax"]

    with pipeline_step("p5b_standardise"):
        X_raw = df[SCORE_COLS].values
        X_std = StandardScaler().fit_transform(X_raw)

    with pipeline_step("p5b_fit_all_k"):
        results = fit_all_k(X_std)

    with pipeline_step("p5b_figures"):
        fig_validity_indices(results)
        fig_country_heatmap(df, results)
        fig_score_profiles(df, results, mm_cols)
        fig_silhouette_by_k(X_std, results)
        fig_k3_split(df, results, SCORE_COLS)

    with pipeline_step("p5b_summary_csv"):
        rows = []
        for k, r in sorted(results.items()):
            lbs = r["labels"]
            for cid in range(k):
                mask = lbs == cid
                sub  = df[mask]
                mm   = sub[mm_cols].mean()
                rows.append({
                    "k": k, "cluster": cid, "n": int(mask.sum()),
                    "silhouette_global": round(r["silhouette"], 4),
                    "ari_ward":          round(r["ari_ward"], 4),
                    "mean_sil_sample":   round(r["sil_samples"][mask].mean(), 4),
                    "n_negative_sil":    int((r["sil_samples"][mask] < 0).sum()),
                    "score_cost_mm":     round(mm["score_cost_minmax"], 3),
                    "score_talent_mm":   round(mm["score_talent_minmax"], 3),
                    "score_infra_mm":    round(mm["score_infra_minmax"], 3),
                    "score_cluster_mm":  round(mm["score_cluster_minmax"], 3),
                    "top_country":       sub["country_code"].value_counts().index[0],
                    "suggested_label":   dominant_label(
                        pd.Series({
                            "score_cost_minmax_mean":    mm["score_cost_minmax"],
                            "score_talent_minmax_mean":  mm["score_talent_minmax"],
                            "score_infra_minmax_mean":   mm["score_infra_minmax"],
                            "score_cluster_minmax_mean": mm["score_cluster_minmax"],
                        })
                    ),
                })
        summary = pd.DataFrame(rows)
        path = ANA_DIR / "p5b_cluster_summary.csv"
        summary.to_csv(path, index=False)
        log.info("summary saved", path=str(path))

    # ---- Console report -----------------------------------------------------
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS SUMMARY (updated scores -- hi_tech excluded)")
    print("=" * 70)
    print(f"\n{'k':>3}  {'Silhouette':>12}  {'CH index':>10}  "
          f"{'DB index':>10}  {'ARI(Ward)':>10}  Cluster sizes")
    print("-" * 70)
    for k in sorted(results):
        r    = results[k]
        lbs  = r["labels"]
        sizes = "/".join(str(s) for s in np.bincount(lbs))
        print(f"{k:>3}  {r['silhouette']:>12.4f}  {r['ch']:>10.1f}  "
              f"{r['db']:>10.4f}  {r['ari_ward']:>10.4f}  {sizes}")

    print("\n  k=2 vs k=3 silhouette gap: "
          f"{results[2]['silhouette'] - results[3]['silhouette']:+.4f}")
    print("  (negative means k=3 is better; positive means k=2 is better)\n")

    print("Centroid profiles (normalised [0,1] scores):")
    for k in [2, 3, 4]:
        if k not in results:
            continue
        print(f"\n  k={k} --")
        lbs = results[k]["labels"]
        for cid in range(k):
            mask = lbs == cid
            mm   = df.loc[mask, mm_cols].mean()
            top3 = df.loc[mask, "country_code"].value_counts().head(3)
            top3_str = ", ".join(f"{COUNTRY_NAMES.get(c,c)}({n})"
                                 for c, n in top3.items())
            n_trans = int(df.loc[mask, "is_transition_region"].sum())
            print(f"    C{cid+1} (n={mask.sum():3d}, {n_trans} transition): "
                  f"cost={mm['score_cost_minmax']:.2f}  "
                  f"talent={mm['score_talent_minmax']:.2f}  "
                  f"infra={mm['score_infra_minmax']:.2f}  "
                  f"cluster={mm['score_cluster_minmax']:.2f}  "
                  f"| {top3_str}")

    print("\n" + "=" * 70)
    print("Figures written to figures/:")
    for fn in ["p5b_validity_indices.png","p5b_country_heatmap.png",
               "p5b_score_profiles.png","p5b_silhouette_by_k.png",
               "p5b_k3_detail.png"]:
        print(f"  {fn}")
    print("=" * 70)


if __name__ == "__main__":
    main()
