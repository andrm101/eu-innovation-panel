"""
Phase 5 — Clustering & Archetype Assignment
Reads Gold, clusters 242 NUTS2 regions on 4 z-scored dimension scores,
assigns archetype labels, and writes updated Gold parquet.

Clustering approach:
  1. Re-standardise the 4 dimension scores to unit variance (equal weighting).
  2. K-Means (k=2..5), n_init=50, max_iter=500, random_state=42.
  3. Select k by silhouette score; flag if best silhouette < 0.35.
  4. Agglomerative Clustering (Ward) at optimal k as robustness check → report ARI.
  5. Per-region: silhouette_sample, Euclidean distance_to_centroid.
  6. Archetype labels assigned from centroid profile (data-driven, descriptive).

No NUTS2 shapefile present → geographic map deferred to Phase 7 (Dashboard).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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
FIG_DIR.mkdir(exist_ok=True)

SCORE_COLS = ["score_cost", "score_talent", "score_infra", "score_cluster"]
MM_COLS    = ["score_cost_minmax", "score_talent_minmax",
              "score_infra_minmax", "score_cluster_minmax"]

DIM_LABELS = {
    "score_cost":    "Prosperity (cost)",
    "score_talent":  "Talent",
    "score_infra":   "Digital infra.",
    "score_cluster": "Innovation cluster",
}

SILHOUETTE_THRESHOLD = 0.35
K_RANGE              = range(2, 6)

ARCHETYPE_PALETTE = [
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2",
]

_LABEL_TEMPLATES = {
    ("cluster", "high"):   "Frontier R&D clusters",
    ("cluster", "mid"):    "Established knowledge economies",
    ("talent",  "high"):   "Human capital hubs",
    ("talent",  "mid"):    "Talent-intensive service economies",
    ("cost",    "high"):   "High-income diversified regions",
    ("infra",   "high"):   "Digital infrastructure-intensive regions",
    ("none",    "high"):   "Advanced innovation regions",
    ("none",    "low"):    "Catching-up & peripheral regions",
    ("none",    "mid"):    "Diversified mid-range economies",
}

# Threshold below which no single dimension is considered dominant
_BALANCE_THRESHOLD = 0.12


def assign_labels(mm_means: np.ndarray) -> list[str]:
    """Assign descriptive archetype labels based on minmax centroid profiles.

    When no single dimension leads by more than _BALANCE_THRESHOLD the cluster
    is treated as balanced ('none') and labelled by overall level only.
    """
    n_clusters = len(mm_means)
    dim_names  = ["cost", "talent", "infra", "cluster"]
    levels     = mm_means.mean(axis=1)
    # Use 'none' as dominant when dimension spread is too narrow to be meaningful
    dominant = [
        "none" if (row.max() - row.min()) < _BALANCE_THRESHOLD
        else dim_names[np.argmax(row)]
        for row in mm_means
    ]

    labels, used = [], set()
    for i in range(n_clusters):
        dom = dominant[i]
        lvl = "high" if levels[i] >= 0.55 else ("low" if levels[i] < 0.40 else "mid")

        label = _LABEL_TEMPLATES.get((dom, lvl))

        if label is None or label in used:
            if lvl == "high":
                label = f"High-performance {'innovation' if dom == 'cluster' else dom} regions"
            elif lvl == "low":
                label = "Catching-up & peripheral regions"
            else:
                label = f"Intermediate {dom}-oriented regions"

        if label in used:
            label = f"{label} (II)"
        used.add(label)
        labels.append(label)

    return labels


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_silhouette_search(metrics: dict) -> None:
    ks  = sorted(metrics.keys())
    sil = [metrics[k]["silhouette"] for k in ks]
    ch  = [metrics[k]["calinski_harabasz"] for k in ks]
    db  = [metrics[k]["davies_bouldin"] for k in ks]
    best_k = ks[int(np.argmax(sil))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, vals, ylabel, title, color in zip(
        axes,
        [sil, ch, db],
        ["Silhouette score",
         "Calinski-Harabász index",
         "Davies-Bouldin index (lower = more separated)"],
        ["Silhouette vs k", "Calinski-Harabász vs k", "Davies-Bouldin vs k"],
        ["#4c72b0", "#55a868", "#dd8452"],
    ):
        ax.plot(ks, vals, "o-", color=color, linewidth=2)
        ax.axvline(best_k, color="#c44e52", linestyle=":", linewidth=1.2,
                   label=f"Selected k={best_k}")
        if ylabel.startswith("Silhouette"):
            ax.axhline(SILHOUETTE_THRESHOLD, color="grey", linestyle="--",
                       linewidth=0.9, label=f"Threshold ({SILHOUETTE_THRESHOLD})")
        ax.set_xlabel("k (number of archetypes)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(ks)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Cluster validity indices across k  |  EU NUTS2 Regional Innovation Panel\n"
        "Red dotted: selected k",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p5_silhouette_search.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_cluster_scatter(df: pd.DataFrame, labels: list[str],
                         palette: list[str]) -> None:
    pairs = [
        ("score_cost",    "score_cluster", "Prosperity", "Innovation cluster"),
        ("score_talent",  "score_cluster", "Talent",     "Innovation cluster"),
        ("score_infra",   "score_cluster", "Digital infra.", "Innovation cluster"),
        ("score_cost",    "score_talent",  "Prosperity", "Talent"),
        ("score_cost",    "score_infra",   "Prosperity", "Digital infra."),
        ("score_talent",  "score_infra",   "Talent",     "Digital infra."),
    ]
    n_archetypes = len(labels)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, (xcol, ycol, xlbl, ylbl) in zip(axes.flatten(), pairs):
        for aid in range(n_archetypes):
            mask = df["archetype_id"].values == aid
            ax.scatter(df.loc[df.index[mask], xcol],
                       df.loc[df.index[mask], ycol],
                       s=14, alpha=0.65, color=palette[aid],
                       label=labels[aid], zorder=2 + aid)
        ax.set_xlabel(xlbl, fontsize=9)
        ax.set_ylabel(ylbl, fontsize=9)
        ax.axhline(0, color="lightgrey", linewidth=0.5, zorder=0)
        ax.axvline(0, color="lightgrey", linewidth=0.5, zorder=0)
        ax.tick_params(labelsize=7)

    patches = [mpatches.Patch(color=palette[i], label=f"A{i+1}: {labels[i]}")
               for i in range(n_archetypes)]
    fig.legend(handles=patches, loc="lower center", ncol=min(3, n_archetypes),
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Pairwise scatter of dimension scores by archetype  |  EU NUTS2\n"
        "Source: Eurostat; K-Means clustering on z-scaled scores",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = FIG_DIR / "p5_cluster_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_radar(df: pd.DataFrame, labels: list[str], palette: list[str]) -> None:
    n_archetypes = len(labels)
    dim_lbls = ["Prosperity\n(cost)", "Talent", "Digital\ninfra.", "Innovation\ncluster"]
    N = 4
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    means = []
    for aid in range(n_archetypes):
        mask = df["archetype_id"] == aid
        means.append(df.loc[mask, MM_COLS].mean().values)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for aid, (m, lbl) in enumerate(zip(means, labels)):
        values = m.tolist() + [m[0]]
        ax.plot(angles, values, "o-", linewidth=1.8,
                color=palette[aid], label=f"A{aid+1}: {lbl}")
        ax.fill(angles, values, alpha=0.12, color=palette[aid])

    ax.set_thetagrids(np.degrees(angles[:-1]), dim_lbls, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.15), fontsize=8.5)
    ax.set_title(
        "Mean normalised dimension scores by archetype\n"
        "EU NUTS2 Regional Innovation Panel  |  Source: Eurostat",
        pad=20, fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p5_archetype_radar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_archetype_profiles(df: pd.DataFrame, labels: list[str],
                            palette: list[str]) -> None:
    n_archetypes = len(labels)
    mat = np.zeros((n_archetypes, 4))
    for aid in range(n_archetypes):
        mask = df["archetype_id"] == aid
        mat[aid] = df.loc[mask, SCORE_COLS].mean().values

    counts = [int((df["archetype_id"] == aid).sum()) for aid in range(n_archetypes)]
    row_lbls = [f"A{aid+1}: {labels[aid]} (n={counts[aid]})"
                for aid in range(n_archetypes)]
    col_lbls = [DIM_LABELS[c] for c in SCORE_COLS]

    fig, ax = plt.subplots(figsize=(10, 1.5 * n_archetypes + 1.5))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(col_lbls, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(n_archetypes))
    ax.set_yticklabels(row_lbls, fontsize=9)
    for i in range(n_archetypes):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean z-score")
    ax.set_title(
        "Mean dimension z-scores per archetype  |  EU NUTS2\n"
        "Source: Eurostat",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p5_archetype_profiles.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_silhouette_distribution(df: pd.DataFrame, labels: list[str],
                                 palette: list[str]) -> None:
    n_archetypes = len(labels)
    fig, ax = plt.subplots(figsize=(12, 5))
    y_lower = 10
    for aid in range(n_archetypes):
        mask = df["archetype_id"] == aid
        sil_vals = df.loc[mask, "silhouette_sample"].sort_values().values
        y_upper = y_lower + len(sil_vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_vals,
                         facecolor=palette[aid], alpha=0.8)
        ax.text(-0.03, (y_lower + y_upper) / 2, f"A{aid+1}",
                ha="right", va="center", fontsize=8, color=palette[aid])
        y_lower = y_upper + 5

    mean_sil = df["silhouette_sample"].mean()
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(mean_sil, color="red", linewidth=1.2, linestyle="--",
               label=f"Mean silhouette = {mean_sil:.3f}")
    ax.set_xlabel("Silhouette coefficient", fontsize=10)
    ax.set_ylabel("NUTS2 region (sorted within archetype)", fontsize=10)
    ax.set_title(
        "Per-region silhouette coefficients by archetype  |  EU NUTS2\n"
        "Negative values indicate possible misclassification  |  Source: Eurostat",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_yticks([])
    plt.tight_layout()
    path = FIG_DIR / "p5_silhouette_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


def plot_country_archetype(df: pd.DataFrame, labels: list[str],
                           palette: list[str]) -> None:
    n_archetypes = len(labels)
    counts = (
        df.groupby(["country_code", "archetype_id"])
        .size()
        .unstack(fill_value=0)
    )
    # Sort by share of lowest archetype id (typically catching-up)
    # Use mean archetype id per country as sort key (higher = more innovation-intensive)
    mean_id = (counts * np.arange(n_archetypes)).sum(axis=1) / counts.sum(axis=1)
    counts = counts.loc[mean_id.sort_values().index]
    fracs  = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(fracs))
    for aid in range(n_archetypes):
        if aid in fracs.columns:
            vals = fracs[aid].values
            ax.bar(range(len(fracs)), vals, bottom=bottom, color=palette[aid],
                   label=f"A{aid+1}: {labels[aid]}", alpha=0.9)
            bottom += vals

    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(fracs.index.tolist(), rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Proportion of NUTS2 regions", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.0), fontsize=8)
    ax.set_title(
        "Archetype composition by country (proportion of NUTS2 regions)\n"
        "EU NUTS2 Regional Innovation Panel  |  Source: Eurostat",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "p5_country_archetype.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure", path=str(path))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    with pipeline_step("p5_load_gold"):
        df = pd.read_parquet(GOLD_PATH)
        log.info("gold loaded", shape=df.shape)

    with pipeline_step("p5_prepare_features"):
        X_raw = df[SCORE_COLS].values
        scaler = StandardScaler()
        X_std  = scaler.fit_transform(X_raw)
        log.info("scores re-standardised",
                 pre_std=np.std(X_raw, axis=0).round(3).tolist(),
                 post_std=np.std(X_std, axis=0).round(3).tolist())

    with pipeline_step("p5_silhouette_search"):
        metrics = {}
        for k in K_RANGE:
            km  = KMeans(n_clusters=k, n_init=50, max_iter=500, random_state=42)
            lbs = km.fit_predict(X_std)
            metrics[k] = {
                "silhouette":         silhouette_score(X_std, lbs),
                "calinski_harabasz":  calinski_harabasz_score(X_std, lbs),
                "davies_bouldin":     davies_bouldin_score(X_std, lbs),
            }
            log.info("k-means metric", k=k,
                     silhouette=round(metrics[k]["silhouette"], 4),
                     calinski_harabasz=round(metrics[k]["calinski_harabasz"], 1),
                     davies_bouldin=round(metrics[k]["davies_bouldin"], 4))

        best_k   = max(metrics, key=lambda k_: metrics[k_]["silhouette"])
        best_sil = metrics[best_k]["silhouette"]
        log.info("optimal k selected", k=best_k, silhouette=round(best_sil, 4))

        if best_sil < SILHOUETTE_THRESHOLD:
            log.warning(
                "best silhouette below threshold; human review recommended",
                best_silhouette=round(best_sil, 4),
                threshold=SILHOUETTE_THRESHOLD,
            )

    with pipeline_step("p5_fit_kmeans"):
        km_final  = KMeans(n_clusters=best_k, n_init=50, max_iter=500, random_state=42)
        km_labels = km_final.fit_predict(X_std)
        log.info("K-Means fitted", k=best_k,
                 inertia=round(km_final.inertia_, 2),
                 cluster_sizes=np.bincount(km_labels).tolist())

    with pipeline_step("p5_robustness_ward"):
        ward_labels = AgglomerativeClustering(
            n_clusters=best_k, linkage="ward"
        ).fit_predict(X_std)
        ari = adjusted_rand_score(km_labels, ward_labels)
        log.info("Ward vs K-Means robustness",
                 adjusted_rand_index=round(ari, 4),
                 interpretation=(
                     "strong agreement" if ari >= 0.8 else
                     "moderate agreement" if ari >= 0.5 else
                     "weak agreement — review cluster solution"
                 ))

    with pipeline_step("p5_per_region_metrics"):
        sil_samples       = silhouette_samples(X_std, km_labels)
        centroids_std     = km_final.cluster_centers_
        dist_to_centroid  = np.array([
            np.linalg.norm(X_std[i] - centroids_std[km_labels[i]])
            for i in range(len(X_std))
        ])
        log.info("per-region metrics",
                 mean_silhouette=round(float(sil_samples.mean()), 4),
                 mean_distance=round(float(dist_to_centroid.mean()), 4),
                 n_negative_silhouette=int((sil_samples < 0).sum()))

    with pipeline_step("p5_archetype_labels"):
        mm_means = np.zeros((best_k, 4))
        for aid in range(best_k):
            mask = km_labels == aid
            mm_means[aid] = df.loc[df.index[mask], MM_COLS].mean().values

        archetype_labels = assign_labels(mm_means)
        for aid, lbl in enumerate(archetype_labels):
            n = int((km_labels == aid).sum())
            log.info("archetype", id=aid, label=lbl, n_regions=n,
                     mm_cost=round(mm_means[aid, 0], 3),
                     mm_talent=round(mm_means[aid, 1], 3),
                     mm_infra=round(mm_means[aid, 2], 3),
                     mm_cluster=round(mm_means[aid, 3], 3))

    with pipeline_step("p5_update_gold"):
        df["archetype_id"]         = km_labels.astype(int)
        df["archetype_label"]      = [archetype_labels[i] for i in km_labels]
        df["distance_to_centroid"] = dist_to_centroid.astype(float)
        df["silhouette_sample"]    = sil_samples.astype(float)

    with pipeline_step("p5_validate_gold"):
        from contracts.gold_schema import gold_schema
        try:
            gold_schema.validate(df, lazy=True)
            log.info("gold schema validation PASSED")
        except Exception as exc:
            log.warning("gold schema validation issues (non-fatal)",
                        error=str(exc)[:400])

    with pipeline_step("p5_write_gold"):
        df.to_parquet(GOLD_PATH, engine="pyarrow", index=True)
        log.info("gold written", path=str(GOLD_PATH), shape=df.shape)

    with pipeline_step("p5_figures"):
        pal = ARCHETYPE_PALETTE[:best_k]
        plot_silhouette_search(metrics)
        plot_cluster_scatter(df, archetype_labels, pal)
        plot_radar(df, archetype_labels, pal)
        plot_archetype_profiles(df, archetype_labels, pal)
        plot_silhouette_distribution(df, archetype_labels, pal)
        plot_country_archetype(df, archetype_labels, pal)

    # ── Gate check ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("P5 GATE CHECK")
    print("=" * 65)
    n_labelled = int(df["archetype_id"].notna().sum())
    print(f"  archetype_id coverage          : {n_labelled}/{len(df)} "
          f"[{'PASS' if n_labelled == len(df) else 'FAIL'}]")
    print(f"  optimal k                      : {best_k}")
    print(f"  best silhouette                : {best_sil:.4f}  "
          f"[{'PASS' if best_sil >= SILHOUETTE_THRESHOLD else 'WARN — below threshold'}]")
    print(f"  Ward ARI vs K-Means            : {ari:.4f}  "
          f"[{'PASS' if ari >= 0.5 else 'WARN — weak agreement'}]")
    print(f"  mean silhouette sample         : {sil_samples.mean():.4f}")
    print(f"  regions with negative sil.     : {int((sil_samples < 0).sum())}")
    print("\n  Archetype composition:")
    for aid, lbl in enumerate(archetype_labels):
        n = int((df["archetype_id"] == aid).sum())
        print(f"    A{aid+1} ({n:3d} regions) : {lbl}")

    all_pass = (n_labelled == len(df)
                and best_sil >= SILHOUETTE_THRESHOLD
                and ari >= 0.5)
    print(f"\n  Gold shape                     : {df.shape}")
    print(f"  Gold path                      : {GOLD_PATH}")
    print("\n" + ("GATE_P5=PASS" if all_pass else "GATE_P5=WARN — review silhouette/ARI"))
    print("=" * 65)


if __name__ == "__main__":
    main()
