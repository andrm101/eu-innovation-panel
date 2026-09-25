"""
Phase 3 — Exploratory Data Analysis

P3.T1  Load Silver + validate shape
P3.T2  Univariate distributions (histogram + KDE, skewness/kurtosis, normality)
P3.T3  Correlation matrix (Pearson + Spearman) + VIF
P3.T4  Country-level boxplots (cross-country variance decomposition)
P3.T5  Spatial autocorrelation — global Moran's I + LISA maps (KNN-8)
P3.T6  Missingness/imputation pattern heatmap
P3.T7  PCA biplot preview (standardised, PC1 x PC2)
P3.T8  Write analysis/eda_report.json
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

np.random.seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import configure_logging, pipeline_step

configure_logging()

FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

DPI = 300
SOURCE_LINE = "Source: Eurostat SDMX-CSV; EU-Innovation-Panel pipeline (2026)"

FEATURE_COLS = [
    "gdp_per_capita_pps",
    "population",
    "hrst_per_1000",
    "tertiary_enrolment_rate",
    "employment_rate",
    "broadband_penetration_pct",
    "rd_expenditure_pct_gdp",
    "business_rd_pct_gdp",
    "epo_patents_per_mio_pop",
    "net_migration_rate",
    "hi_tech_employment_pct",
    "gva_ict_share",
    "population_density",
    "enterprise_internet_use",
    "lq_nace_j62j63",
    "lq_nace_c21_m72",
    "lq_nace_c26",
    "lq_nace_d35_clean",
]

FEATURE_LABELS = {
    "gdp_per_capita_pps":       "GDP/capita (PPS)",
    "population":               "Population (thousands)",
    "hrst_per_1000":            "HRST per 1000 active",
    "tertiary_enrolment_rate":  "Tertiary attainment (%)",
    "employment_rate":          "Employment rate (%)",
    "broadband_penetration_pct":"Broadband penetration (%)",
    "rd_expenditure_pct_gdp":   "GERD (% GDP)",
    "business_rd_pct_gdp":      "BERD (% GDP)",
    "epo_patents_per_mio_pop":  "EPO patents / mio pop",
    "net_migration_rate":       "Net migration rate",
    "hi_tech_employment_pct":   "Hi-tech employment (%)",
    "gva_ict_share":            "GVA ICT share (%)",
    "population_density":       "Pop. density (per km2)",
    "enterprise_internet_use":  "Enterprise internet use (%)",
    "lq_nace_j62j63":           "LQ ICT services (J62/63)",
    "lq_nace_c21_m72":          "LQ KIS hi-tech (proxy)",
    "lq_nace_c26":              "LQ hi-tech mfg (proxy)",
    "lq_nace_d35_clean":        "LQ energy utilities (proxy)",
}

DIMENSION_MAP = {
    "gdp_per_capita_pps":       "cost",
    "population":               "context",
    "hrst_per_1000":            "talent",
    "tertiary_enrolment_rate":  "talent",
    "employment_rate":          "talent",
    "broadband_penetration_pct":"infra",
    "rd_expenditure_pct_gdp":   "cluster",
    "business_rd_pct_gdp":      "cluster",
    "epo_patents_per_mio_pop":  "cluster",
    "net_migration_rate":       "context",
    "hi_tech_employment_pct":   "talent",
    "gva_ict_share":            "cluster",
    "population_density":       "context",
    "enterprise_internet_use":  "infra",
    "lq_nace_j62j63":           "cluster",
    "lq_nace_c21_m72":          "cluster",
    "lq_nace_c26":              "cluster",
    "lq_nace_d35_clean":        "cluster",
}

DIM_COLORS = {
    "cost": "#e41a1c", "talent": "#377eb8",
    "infra": "#4daf4a", "cluster": "#984ea3", "context": "#888888",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_silver() -> pd.DataFrame:
    path = ROOT / "data/silver/region_profiles_silver.parquet"
    df = pd.read_parquet(path)
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def normality_test(series: pd.Series) -> dict:
    """Anderson-Darling normality test (N=242 exceeds Shapiro-Wilk limit)."""
    clean = series.dropna()
    if len(clean) < 8:
        return {"ad_statistic": None, "normal_at_05": None}
    result = stats.anderson(clean, dist="norm")
    normal = bool(result.statistic < result.critical_values[2])  # alpha=0.05
    return {
        "ad_statistic": round(float(result.statistic), 4),
        "ad_critical_05": round(float(result.critical_values[2]), 4),
        "normal_at_05": normal,
    }


# ── P3.T2: Univariate distributions ───────────────────────────────────────────

def plot_distributions(df: pd.DataFrame) -> tuple:
    ncols, nrows = 4, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 22))
    axes_flat = axes.flatten()

    dist_stats = {}
    avail = [f for f in FEATURE_COLS if f in df.columns]
    for i, feat in enumerate(avail):
        ax = axes_flat[i]
        series = df[feat].dropna()
        color = DIM_COLORS[DIMENSION_MAP[feat]]
        skew = float(series.skew())
        kurt = float(series.kurtosis())
        dist_stats[feat] = {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "n_valid": int(series.notna().sum()),
        }
        dist_stats[feat].update(normality_test(series))

        ax.hist(series, bins=30, color=color, alpha=0.5, density=True, edgecolor="none")
        kde_x = np.linspace(series.min(), series.max(), 300)
        try:
            kde = stats.gaussian_kde(series)
            ax.plot(kde_x, kde(kde_x), color=color, lw=1.5)
        except Exception:
            pass
        ax.axvline(series.median(), color="black", lw=1, ls="--", alpha=0.7)
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=8, pad=3)
        ax.set_xlabel(f"skew={skew:.2f}  kurt={kurt:.2f}", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("")

    for j in range(len(avail), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Univariate Distributions — EU27 NUTS2 Regional Features\n"
        "(dashed line = median; colour indicates analytical dimension)",
        fontsize=11, y=1.005,
    )
    patches = [mpatches.Patch(color=v, label=k) for k, v in DIM_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.025, SOURCE_LINE, ha="center", fontsize=6, color="grey")
    return save_fig(fig, "p3_feature_distributions.png"), dist_stats


# ── P3.T3: Correlation matrix + VIF ───────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> tuple:
    avail = [f for f in FEATURE_COLS if f in df.columns]
    X = df[avail].copy()
    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")

    vif_results = {}
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_vif = X.dropna()
        if len(X_vif) > len(avail) + 1:
            vif_vals = [
                variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])
            ]
            vif_results = {k: round(v, 2) for k, v in zip(avail, vif_vals)}
    except Exception as exc:
        vif_results = {"error": str(exc)}

    labels = [FEATURE_LABELS.get(c, c) for c in avail]
    mask_upper = np.triu(np.ones_like(pearson, dtype=bool), k=1)

    for corr_mat, fname, title in [
        (pearson,  "p3_correlation_matrix.png",  "Pearson"),
        (spearman, "p3_spearman_correlation.png", "Spearman Rank"),
    ]:
        fig, ax = plt.subplots(figsize=(15, 13))
        sns.heatmap(
            corr_mat, mask=mask_upper, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.3, linecolor="white",
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 6}, ax=ax,
        )
        ax.set_title(
            f"{title} Correlation Matrix — 18 Silver Features "
            f"(EU27 NUTS2, N={len(X)})\nLower triangle; upper masked",
            fontsize=10, pad=8,
        )
        ax.tick_params(labelsize=7)
        fig.text(0.5, -0.01, SOURCE_LINE, ha="center", fontsize=6, color="grey")
        save_fig(fig, fname)

    high_corr = [
        {
            "feat_a": avail[i], "feat_b": avail[j],
            "pearson_r": round(float(pearson.iloc[i, j]), 3),
            "flag": "multicollinearity_risk" if abs(pearson.iloc[i, j]) > 0.85 else "moderate",
        }
        for i in range(len(avail))
        for j in range(i + 1, len(avail))
        if abs(pearson.iloc[i, j]) > 0.70
    ]

    return {"vif": vif_results, "high_corr_pairs": high_corr}


# ── P3.T4: Country-level boxplots ─────────────────────────────────────────────

def plot_country_boxplots(df: pd.DataFrame) -> Path:
    features_to_plot = [
        "gdp_per_capita_pps", "rd_expenditure_pct_gdp",
        "hrst_per_1000", "lq_nace_j62j63",
        "broadband_penetration_pct", "hi_tech_employment_pct",
    ]
    country_order = (
        df.groupby("country_code")["gdp_per_capita_pps"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    axes_flat = axes.flatten()

    for i, feat in enumerate(features_to_plot):
        ax = axes_flat[i]
        data = df[[feat, "country_code"]].dropna()
        color = DIM_COLORS[DIMENSION_MAP[feat]]
        groups = [
            data[data["country_code"] == cc][feat].values
            for cc in country_order
            if cc in data["country_code"].values
        ]
        group_labels = [
            cc for cc in country_order
            if cc in data["country_code"].values
        ]
        ax.boxplot(
            groups, tick_labels=group_labels, patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.4),
            medianprops=dict(color="black", lw=2),
            flierprops=dict(marker=".", markersize=3, alpha=0.5),
        )
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        "Cross-Country Variation — EU27 NUTS2 Regions\n"
        "(countries ordered by median GDP/capita; boxes = IQR; whiskers = 1.5xIQR)",
        fontsize=11, y=1.005,
    )
    fig.text(0.5, -0.01, SOURCE_LINE, ha="center", fontsize=6, color="grey")
    return save_fig(fig, "p3_country_boxplots.png")


# ── P3.T5: Spatial autocorrelation ────────────────────────────────────────────

def run_spatial_analysis(df: pd.DataFrame, log) -> tuple:
    try:
        import geopandas as gpd
        import libpysal
        from esda.moran import Moran, Moran_Local
    except ImportError as exc:
        log.warning("spatial_skipped", reason=str(exc))
        return None, {}

    geojson_path = ROOT / "data/raw/eurostat/nuts2_2021_geojson.json"
    if not geojson_path.exists():
        log.warning("spatial_skipped", reason="GeoJSON not found")
        return None, {}

    EU27 = {
        "AT","BE","BG","CY","CZ","DE","DK","EE","EL","ES",
        "FI","FR","HR","HU","IE","IT","LT","LU","LV","MT",
        "NL","PL","PT","RO","SE","SI","SK",
    }

    gdf = gpd.read_file(geojson_path)
    gdf = gdf[
        (gdf["NUTS_ID"].str.len() == 4) &
        (gdf["NUTS_ID"].str[:2].isin(EU27))
    ].copy()
    gdf = gdf.rename(columns={"NUTS_ID": "nuts2_code"})

    merged = gdf.merge(df.reset_index(), on="nuts2_code", how="inner")
    if len(merged) < 100:
        log.warning("spatial_skipped", reason=f"Only {len(merged)} regions matched GeoJSON")
        return None, {}

    # KNN-8 avoids island isolation (CY, MT, ES70, FR91-FR94, PT20/PT30)
    w = libpysal.weights.KNN.from_dataframe(merged, k=8)
    w.transform = "r"

    spatial_results = {}
    avail = [f for f in FEATURE_COLS if f in merged.columns]
    for feat in avail:
        y = merged[feat].fillna(merged[feat].median()).values
        mi = Moran(y, w, permutations=999)
        spatial_results[feat] = {
            "morans_i": round(float(mi.I), 4),
            "p_value": round(float(mi.p_sim), 4),
            "z_score": round(float(mi.z_sim), 4),
            "significant_at_05": bool(mi.p_sim < 0.05),
        }

    # LISA maps for 4 key features
    LISA_LABELS = {
        "HH": "High-High", "LL": "Low-Low",
        "LH": "Low-High",  "HL": "High-Low", "NS": "Not significant",
    }
    LISA_COLORS = {
        "HH": "#d7191c", "LL": "#2c7bb6",
        "LH": "#abd9e9", "HL": "#fdae61", "NS": "#eeeeee",
    }
    quadrant_map = {1: "HH", 3: "LL", 4: "LH", 2: "HL"}

    lisa_features = [
        "gdp_per_capita_pps", "rd_expenditure_pct_gdp",
        "lq_nace_j62j63", "hi_tech_employment_pct",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = axes.flatten()

    for i, feat in enumerate([f for f in lisa_features if f in merged.columns]):
        ax = axes_flat[i]
        y = merged[feat].fillna(merged[feat].median()).values
        lisa = Moran_Local(y, w, permutations=999)
        quad = pd.Series(lisa.q).map(quadrant_map).fillna("NS")
        quad_sig = quad.where(lisa.p_sim < 0.05, "NS")
        merged = merged.copy()
        merged["_lisa_quad"] = quad_sig.values

        for label, color in LISA_COLORS.items():
            sub = merged[merged["_lisa_quad"] == label]
            if not sub.empty:
                sub.plot(ax=ax, color=color, linewidth=0.1, edgecolor="white")

        mi_val = spatial_results.get(feat, {}).get("morans_i", float("nan"))
        p_val = spatial_results.get(feat, {}).get("p_value", float("nan"))
        ax.set_title(
            f"{FEATURE_LABELS.get(feat, feat)}\n"
            f"Moran's I = {mi_val:.3f}  (p = {p_val:.3f})",
            fontsize=9,
        )
        ax.set_axis_off()

        if i == 0:
            patches = [
                mpatches.Patch(color=v, label=f"{k} — {LISA_LABELS[k]}")
                for k, v in LISA_COLORS.items()
            ]
            ax.legend(handles=patches, loc="lower left", fontsize=7, framealpha=0.8)

    fig.suptitle(
        "LISA Spatial Clusters — EU27 NUTS2 Regions\n"
        "(KNN-8 weights; alpha = 0.05; 999 permutations)",
        fontsize=12, y=1.005,
    )
    fig.text(0.5, -0.005, SOURCE_LINE, ha="center", fontsize=6, color="grey")
    path_lisa = save_fig(fig, "p3_lisa_maps.png")

    # Moran scatter for feature with highest |I|
    sig_feats = [f for f in spatial_results if spatial_results[f]["significant_at_05"]]
    top_feat = max(
        sig_feats if sig_feats else [lisa_features[0]],
        key=lambda f: abs(spatial_results[f]["morans_i"]),
    )
    y = merged[top_feat].fillna(merged[top_feat].median()).values
    yz = (y - y.mean()) / y.std()
    lag_yz = libpysal.weights.lag_spatial(w, yz)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(yz, lag_yz, s=15, alpha=0.6, color="#555555")
    b, a_int = np.polyfit(yz, lag_yz, 1)
    xs = np.linspace(yz.min(), yz.max(), 100)
    ax2.plot(xs, a_int + b * xs, "r-", lw=2)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.axvline(0, color="black", lw=0.5)
    ax2.set_xlabel(f"Standardised {FEATURE_LABELS.get(top_feat, top_feat)}", fontsize=9)
    ax2.set_ylabel("Spatial lag (KNN-8)", fontsize=9)
    ax2.set_title(
        f"Moran Scatter Plot — {FEATURE_LABELS.get(top_feat, top_feat)}\n"
        f"Moran's I = {spatial_results[top_feat]['morans_i']:.4f}  "
        f"(p = {spatial_results[top_feat]['p_value']:.4f})",
        fontsize=9,
    )
    fig2.text(0.5, -0.01, SOURCE_LINE, ha="center", fontsize=6, color="grey")
    save_fig(fig2, "p3_moran_scatter.png")

    return path_lisa, spatial_results


# ── P3.T6: Missingness / imputation heatmap ───────────────────────────────────

def plot_missingness_heatmap(df: pd.DataFrame) -> Path:
    flag_cols = [c for c in df.columns if c.endswith("_imputed_flag")]
    if not flag_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No imputed flag columns found", ha="center")
        return save_fig(fig, "p3_missingness_heatmap.png")

    clean_labels = [c.replace("_imputed_flag", "") for c in flag_cols]
    matrix = df[flag_cols].astype(float)
    matrix.columns = clean_labels

    n_imp = matrix.sum(axis=1)
    matrix = matrix.loc[n_imp.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, max(6, len(matrix) // 15)))
    sns.heatmap(
        matrix.T, cmap=["#f0f0f0", "#d73027"],
        linewidths=0, linecolor=None,
        yticklabels=True, xticklabels=False,
        cbar_kws={"label": "Imputed", "ticks": [0, 1]},
        ax=ax,
    )
    imputed_pct = matrix.values.mean() * 100
    ax.set_xlabel(
        f"NUTS2 regions (N={len(matrix)}, sorted by imputed count; "
        f"overall {imputed_pct:.1f}% cells imputed)",
        fontsize=8,
    )
    ax.set_ylabel("Feature", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        "Imputation Pattern — Silver Layer\n"
        "(red = value was imputed via MCAR/MAR procedure; grey = observed)",
        fontsize=10, pad=8,
    )
    fig.text(0.5, -0.02, SOURCE_LINE, ha="center", fontsize=6, color="grey")
    return save_fig(fig, "p3_missingness_heatmap.png")


# ── P3.T7: PCA biplot preview ─────────────────────────────────────────────────

def plot_pca_biplot(df: pd.DataFrame) -> tuple:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    avail = [f for f in FEATURE_COLS if f in df.columns]
    X = df[avail].copy()
    X = X.fillna(X.median())
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(10, len(avail)), random_state=42)
    scores = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_
    loadings = pca.components_.T

    countries = df["country_code"].values
    unique_cc = sorted(set(countries))
    cmap = plt.get_cmap("tab20")
    cc_colors = {cc: cmap(i / max(len(unique_cc) - 1, 1)) for i, cc in enumerate(unique_cc)}

    # Scree plot
    fig_scree, ax_scree = plt.subplots(figsize=(7, 4))
    ax_scree.bar(range(1, len(ev) + 1), ev * 100, color="#4daf4a", edgecolor="white")
    ax_scree.plot(range(1, len(ev) + 1), np.cumsum(ev) * 100, "ko-", ms=5, label="Cumulative")
    ax_scree.axhline(80, color="red", lw=1, ls="--", label="80% threshold")
    ax_scree.set_xlabel("Principal Component", fontsize=9)
    ax_scree.set_ylabel("Variance explained (%)", fontsize=9)
    ax_scree.set_title("Scree Plot — 18 Silver Features (Standardised)", fontsize=10)
    ax_scree.legend(fontsize=8)
    fig_scree.text(0.5, -0.01, SOURCE_LINE, ha="center", fontsize=6, color="grey")
    save_fig(fig_scree, "p3_scree_plot.png")

    # Biplot
    fig, ax = plt.subplots(figsize=(12, 10))
    for cc in unique_cc:
        mask = countries == cc
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   color=cc_colors[cc], s=25, alpha=0.7, label=cc)

    scale = min(np.abs(scores[:, :2]).max(axis=0)) * 0.8
    for j, feat in enumerate(avail):
        lx = loadings[j, 0] * scale
        ly = loadings[j, 1] * scale
        ax.annotate(
            "", xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="->",
                color=DIM_COLORS[DIMENSION_MAP.get(feat, "context")],
                lw=1.5, alpha=0.8,
            ),
        )
        ax.text(lx * 1.12, ly * 1.12, FEATURE_LABELS.get(feat, feat),
                fontsize=6.5, color=DIM_COLORS[DIMENSION_MAP.get(feat, "context")],
                ha="center", va="center")

    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)", fontsize=10)
    n_80 = int(np.searchsorted(np.cumsum(ev), 0.80)) + 1
    ax.set_title(
        f"PCA Biplot Preview — EU27 NUTS2 Regions\n"
        f"PC1+PC2 = {ev[0]+ev[1]:.1%} of variance; "
        f"{n_80} components needed for 80%",
        fontsize=10, pad=8,
    )
    handles = [mpatches.Patch(color=cc_colors[cc], label=cc) for cc in unique_cc]
    ax.legend(handles=handles, ncol=4, fontsize=7, loc="lower right",
              framealpha=0.7, title="Country", title_fontsize=7)
    dim_patches = [mpatches.Patch(color=v, label=k) for k, v in DIM_COLORS.items()]
    fig.legend(handles=dim_patches, loc="upper left", fontsize=7,
               title="Dimension", title_fontsize=7, framealpha=0.7)
    fig.text(0.5, -0.01, SOURCE_LINE, ha="center", fontsize=6, color="grey")
    path = save_fig(fig, "p3_pca_biplot.png")

    return path, {
        "explained_variance": [round(float(v), 4) for v in ev],
        "cumulative_variance": [round(float(v), 4) for v in np.cumsum(ev)],
        "n_components_80pct": n_80,
        "pc1_top_loadings": sorted(
            [(avail[j], round(float(loadings[j, 0]), 3)) for j in range(len(avail))],
            key=lambda x: abs(x[1]), reverse=True,
        )[:5],
        "pc2_top_loadings": sorted(
            [(avail[j], round(float(loadings[j, 1]), 3)) for j in range(len(avail))],
            key=lambda x: abs(x[1]), reverse=True,
        )[:5],
    }


# ── Quality gate ──────────────────────────────────────────────────────────────

def run_quality_gate(eda_report: dict) -> bool:
    figures = list(FIGURES.glob("p3_*.png"))
    assert len(figures) >= 6, f"Expected >=6 figures, found {len(figures)}"

    n_sig = sum(
        1 for v in eda_report.get("spatial_autocorrelation", {}).values()
        if isinstance(v, dict) and v.get("significant_at_05")
    )
    n_high = len(eda_report.get("correlation", {}).get("high_corr_pairs", []))
    n_comp = eda_report.get("pca_preview", {}).get("n_components_80pct", "?")

    print(f"  Figures written: {len(figures)}")
    print(f"  Features with significant spatial autocorrelation (p<0.05): {n_sig}")
    print(f"  High-correlation pairs (|r|>0.70): {n_high}")
    print(f"  PCA components for 80% variance: {n_comp}")
    print("\n  Methodological notes for manuscript:")
    if eda_report.get("distributions"):
        non_normal = [f for f, s in eda_report["distributions"].items()
                      if s.get("normal_at_05") is False]
        skewed = [f for f, s in eda_report["distributions"].items()
                  if abs(s.get("skewness", 0)) > 1.0]
        print(f"  - Non-normal features (Anderson-Darling): {len(non_normal)}")
        print(f"  - Highly skewed (|skew|>1.0): {skewed[:4]}...")
        print(f"    => Consider log-transform for skewed features before PCA (Phase 4)")
    print("\n  GATE_P3=PASS")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with pipeline_step(
        "P3", random_seed=42,
        input_artifact=ROOT / "data/silver/region_profiles_silver.parquet",
    ) as log:
        print("\n=== Phase 3: Exploratory Data Analysis ===\n")

        print("P3.T1  Loading Silver parquet...")
        df = load_silver()
        print(f"  Shape: {df.shape}")
        assert df.shape[0] >= 200
        assert df.shape[1] >= 20
        log.info("silver_loaded", shape=str(df.shape))

        print("P3.T2  Plotting univariate distributions...")
        path_dist, dist_stats = plot_distributions(df)
        print(f"  Saved: {path_dist.name}")
        skewed = [f for f, s in dist_stats.items() if abs(s["skewness"]) > 1.0]
        non_normal = [f for f, s in dist_stats.items() if s.get("normal_at_05") is False]
        print(f"  Highly skewed (|skew|>1): {skewed}")
        print(f"  Non-normal features: {len(non_normal)}/{len(dist_stats)}")
        log.info("distributions_plotted",
                 skewed=skewed, non_normal_count=len(non_normal))

        print("P3.T3  Computing correlation matrix and VIF...")
        corr_stats = plot_correlation_matrix(df)
        high_corr = corr_stats.get("high_corr_pairs", [])
        vif = corr_stats.get("vif", {})
        high_vif = {k: v for k, v in vif.items()
                    if isinstance(v, float) and v > 5}
        print(f"  High-correlation pairs (|r|>0.70): {len(high_corr)}")
        for p in high_corr:
            print(f"    {p['feat_a']} x {p['feat_b']}: r={p['pearson_r']}  [{p['flag']}]")
        print(f"  VIF > 5: {list(high_vif.keys()) or 'none'}")
        log.info("correlation_computed",
                 high_corr_count=len(high_corr), high_vif_features=list(high_vif.keys()))

        print("P3.T4  Plotting country-level boxplots...")
        path_boxes = plot_country_boxplots(df)
        print(f"  Saved: {path_boxes.name}")

        print("P3.T5  Running spatial autocorrelation (Moran's I + LISA)...")
        path_lisa, spatial_stats = run_spatial_analysis(df, log)
        if spatial_stats:
            sig = {f: v for f, v in spatial_stats.items() if v.get("significant_at_05")}
            print(f"  Significant Moran's I (p<0.05): {len(sig)}/{len(spatial_stats)} features")
            top5 = sorted(sig, key=lambda f: abs(sig[f]["morans_i"]), reverse=True)[:5]
            for f in top5:
                print(f"    {f}: I={sig[f]['morans_i']:.4f}  p={sig[f]['p_value']:.4f}")
        log.info("moran_computed",
                 n_significant=sum(1 for v in spatial_stats.values()
                                   if isinstance(v, dict) and v.get("significant_at_05")))

        print("P3.T6  Plotting missingness/imputation heatmap...")
        path_miss = plot_missingness_heatmap(df)
        print(f"  Saved: {path_miss.name}")

        print("P3.T7  Generating PCA biplot preview...")
        path_pca, pca_stats = plot_pca_biplot(df)
        print(f"  Saved: {path_pca.name}")
        print(f"  PC1: {pca_stats['explained_variance'][0]:.1%}  "
              f"PC2: {pca_stats['explained_variance'][1]:.1%}")
        print(f"  Components for 80% variance: {pca_stats['n_components_80pct']}")
        print(f"  PC1 top loadings: "
              f"{[(f, v) for f, v in pca_stats['pc1_top_loadings'][:3]]}")

        print("P3.T8  Writing EDA report...")
        eda_report = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "silver_shape": list(df.shape),
            "feature_count": len([f for f in FEATURE_COLS if f in df.columns]),
            "distributions": dist_stats,
            "correlation": corr_stats,
            "spatial_autocorrelation": spatial_stats,
            "pca_preview": pca_stats,
        }
        (ROOT / "analysis/eda_report.json").write_text(
            json.dumps(eda_report, indent=2, default=str), encoding="utf-8"
        )

        print("\nP3.T9  Running quality gate...")
        run_quality_gate(eda_report)


if __name__ == "__main__":
    main()
