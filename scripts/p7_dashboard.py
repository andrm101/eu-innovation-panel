"""
Phase 7 — Interactive Dashboard
Streamlit app visualising EU NUTS2 regional innovation archetypes.

Run from project root:
    streamlit run scripts/p7_dashboard.py

No live API calls. All data loaded from data/gold/ and analysis/.
GeoJSON: data/raw/eurostat/nuts2_2021_geojson.json (334 NUTS2 features).

Tabs:
  1. Overview        — archetype summary, country composition, NUTS2 choropleth
  2. Region Explorer — per-region radar, scores, feature table
  3. Score Analysis  — scatter matrix, violin distributions
  4. Barriers & Enablers — Cohen's d forest plot, within-archetype CV
  5. Data Quality    — missingness, imputation flags, transition regions
"""

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

# ── Palantir dark theme ────────────────────────────────────────────────────────

_BG0      = "#060b18"   # deepest background
_BG1      = "#0a0f1e"   # main background
_BG2      = "#0d1529"   # panel / card
_BG3      = "#111d35"   # elevated surface
_BG4      = "#162040"   # hover
_BORDER0  = "#0f1e3a"
_BORDER1  = "#1a2d4a"
_BORDER2  = "#1e3a5f"
_TEXT0    = "#f0f4ff"
_TEXT1    = "#c0d0e8"
_TEXT2    = "#8fa8c8"
_TEXT3    = "#4a6285"
_BLUE     = "#00a8ff"
_AMBER    = "#f5a623"
_GREEN    = "#00cc7a"
_RED      = "#ff3355"

_PALANTIR_COLORWAY = [
    "#00a8ff", "#ff8c42", "#00cc7a", "#ff3355",
    "#a855f7", "#f5a623", "#66d9e8", "#fbbf24",
]

pio.templates["palantir"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=_BG1,
        plot_bgcolor=_BG2,
        font=dict(color=_TEXT1, family="'JetBrains Mono','Courier New',monospace", size=11),
        title=dict(font=dict(color=_TEXT0, size=12), x=0.0, xanchor="left"),
        colorway=_PALANTIR_COLORWAY,
        xaxis=dict(
            gridcolor=_BORDER1, zerolinecolor=_BORDER2, linecolor=_BORDER1,
            tickcolor=_TEXT3, tickfont=dict(color=_TEXT2, size=10),
            title_font=dict(color=_TEXT2, size=11),
        ),
        yaxis=dict(
            gridcolor=_BORDER1, zerolinecolor=_BORDER2, linecolor=_BORDER1,
            tickcolor=_TEXT3, tickfont=dict(color=_TEXT2, size=10),
            title_font=dict(color=_TEXT2, size=11),
        ),
        legend=dict(
            bgcolor=_BG2, bordercolor=_BORDER1, borderwidth=1,
            font=dict(color=_TEXT1, size=10),
        ),
        hoverlabel=dict(
            bgcolor=_BG2, bordercolor=_BORDER2,
            font=dict(color=_TEXT0, family="'JetBrains Mono','Courier New',monospace", size=11),
        ),
        coloraxis=dict(colorbar=dict(
            tickcolor=_TEXT3, tickfont=dict(color=_TEXT2, size=9),
            title_font=dict(color=_TEXT2, size=10),
            bgcolor=_BG2, outlinecolor=_BORDER1, outlinewidth=1,
            len=0.65, thickness=10,
        )),
        polar=dict(
            bgcolor=_BG2,
            radialaxis=dict(gridcolor=_BORDER1, linecolor=_BORDER1,
                            tickfont=dict(color=_TEXT2, size=9)),
            angularaxis=dict(gridcolor=_BORDER1, linecolor=_BORDER1,
                             tickfont=dict(color=_TEXT2, size=10)),
        ),
        geo=dict(
            bgcolor=_BG0, lakecolor=_BG0, landcolor="#0d1a2e",
            showland=True, showlakes=True,
            showcoastlines=True, coastlinecolor=_BORDER2,
            showframe=False,
        ),
    )
)
pio.templates.default = "palantir"

# ── Global CSS ─────────────────────────────────────────────────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg0:#060b18; --bg1:#0a0f1e; --bg2:#0d1529; --bg3:#111d35; --bg4:#162040;
    --b0:#0f1e3a;  --b1:#1a2d4a;  --b2:#1e3a5f;  --b3:#00a8ff33;
    --t0:#f0f4ff;  --t1:#c0d0e8;  --t2:#8fa8c8;  --t3:#4a6285;
    --blue:#00a8ff; --amber:#f5a623; --green:#00cc7a;
    --red:#ff3355;  --cyan:#66d9e8;
    --mono:'JetBrains Mono','Courier New',monospace;
    --sans:'Inter','Segoe UI',system-ui,sans-serif;
    --r:2px;
}

/* ── App shell ── */
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main {
    background-color:var(--bg1) !important;
    font-family:var(--sans);
    color:var(--t0);
}
[data-testid="stHeader"],[data-testid="stToolbar"] {
    background-color:var(--bg0) !important;
    border-bottom:1px solid var(--b1) !important;
}
[data-testid="stDecoration"] { display:none !important; }
[data-testid="stMainBlockContainer"] { padding-top:1.2rem !important; }

/* ── Typography ── */
h1 {
    font-family:var(--mono) !important;
    font-size:1.1rem !important; font-weight:500 !important;
    letter-spacing:0.12em !important; color:var(--blue) !important;
    text-transform:uppercase; border-bottom:1px solid var(--b1);
    padding-bottom:0.4rem; margin-bottom:0.8rem !important;
}
h2,h3 {
    font-family:var(--mono) !important; font-weight:500 !important;
    color:var(--t0) !important; letter-spacing:0.04em;
    font-size:0.85rem !important; text-transform:uppercase !important;
    border-bottom:1px solid var(--b0) !important;
    padding-bottom:0.25rem !important; margin-bottom:0.6rem !important;
}
p,li { color:var(--t1) !important; font-size:0.87rem !important; line-height:1.65; }
.stCaption,[data-testid="stCaptionContainer"] p {
    color:var(--t3) !important; font-size:0.72rem !important;
    font-family:var(--mono) !important; letter-spacing:0.03em;
}
hr { border:none !important; border-top:1px solid var(--b1) !important; margin:0.8rem 0 !important; }
.stMarkdown strong { color:var(--t0) !important; font-weight:600 !important; }
.stMarkdown code {
    background:var(--bg3) !important; color:var(--blue) !important;
    font-family:var(--mono) !important; font-size:0.79rem !important;
    padding:0.1em 0.35em !important; border-radius:var(--r) !important;
    border:1px solid var(--b1) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:var(--bg0) !important;
    border-bottom:1px solid var(--b1) !important; gap:0 !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:var(--t3) !important;
    font-family:var(--mono) !important; font-size:0.68rem !important;
    font-weight:500 !important; letter-spacing:0.1em !important;
    text-transform:uppercase !important;
    border:none !important; border-bottom:2px solid transparent !important;
    padding:0.55rem 1.1rem !important; transition:all 0.12s ease;
}
.stTabs [aria-selected="true"] {
    color:var(--blue) !important; border-bottom-color:var(--blue) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color:var(--t0) !important; background:var(--bg3) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background:var(--bg1) !important; padding-top:0.8rem !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background:var(--bg2) !important; border:1px solid var(--b1) !important;
    border-radius:var(--r) !important; padding:0.6rem 0.9rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color:var(--t3) !important; font-family:var(--mono) !important;
    font-size:0.63rem !important; font-weight:500 !important;
    letter-spacing:0.12em !important; text-transform:uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:var(--t0) !important; font-family:var(--mono) !important;
    font-size:1.4rem !important; font-weight:400 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background:var(--bg2) !important; border:1px solid var(--b1) !important;
    border-radius:var(--r) !important;
}
[data-testid="stDataFrame"] iframe { background:var(--bg2) !important; }

/* ── Selects ── */
[data-testid="stSelectbox"] label p,[data-testid="stMultiSelect"] label p {
    color:var(--t3) !important; font-family:var(--mono) !important;
    font-size:0.63rem !important; letter-spacing:0.1em !important;
    text-transform:uppercase !important;
}
[data-baseweb="select"] {
    background:var(--bg2) !important; border:1px solid var(--b1) !important;
    border-radius:var(--r) !important;
}
[data-baseweb="select"] [data-baseweb="select-dropdown"] {
    background:var(--bg2) !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div { color:var(--t1) !important; font-family:var(--mono) !important; font-size:0.81rem !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background:var(--bg2) !important; border:1px solid var(--b1) !important;
    border-radius:var(--r) !important;
}
[data-testid="stExpander"] details summary p {
    color:var(--t2) !important; font-family:var(--mono) !important;
    font-size:0.72rem !important; letter-spacing:0.08em !important;
    text-transform:uppercase !important;
}

/* ── Info / alerts ── */
[data-testid="stInfo"] {
    background:var(--bg2) !important; border-left:3px solid var(--blue) !important;
    border-radius:var(--r) !important;
}
[data-testid="stInfo"] p { color:var(--t1) !important; }

/* ── Plotly chart wrappers ── */
[data-testid="stPlotlyChart"] {
    border:1px solid var(--b1) !important; border-radius:var(--r) !important;
    background:transparent !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width:3px; height:3px; }
::-webkit-scrollbar-track { background:var(--bg0); }
::-webkit-scrollbar-thumb { background:var(--b2); border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:var(--blue); }
</style>
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
GOLD_PATH    = ROOT / "data" / "gold" / "region_profiles_gold.parquet"
P6_EFF       = ROOT / "analysis" / "p6_effect_sizes.csv"
P6D_PATTERNS = ROOT / "analysis" / "p6d_cospec_patterns.csv"
P6D_SUMMARY  = ROOT / "analysis" / "p6d_cospec_summary.csv"
GEOJSON_PATH = ROOT / "data" / "raw" / "eurostat" / "nuts2_2021_geojson.json"

# ── Constants ─────────────────────────────────────────────────────────────────
SCORE_COLS = ["score_cost", "score_talent", "score_infra", "score_cluster"]
MM_COLS    = ["score_cost_minmax", "score_talent_minmax",
              "score_infra_minmax", "score_cluster_minmax"]
DIM_LABELS = ["Prosperity (cost)", "Talent", "Digital infra.", "Innovation cluster"]

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
FEATURE_COLS = list(FEATURE_LABELS.keys())

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

ARCHETYPE_COLORS = {
    "Catching-up & peripheral regions": "#4c9be8",   # bright blue — readable on dark
    "Advanced innovation regions":      "#ff8c42",   # bright orange — readable on dark
}
ARCHETYPE_SHORT = {
    "Catching-up & peripheral regions": "A1: Catching-up",
    "Advanced innovation regions":      "A2: Advanced",
}

COUNTRY_NAMES = {
    "AT": "Austria",    "BE": "Belgium",   "BG": "Bulgaria",  "CY": "Cyprus",
    "CZ": "Czechia",    "DE": "Germany",   "DK": "Denmark",   "EE": "Estonia",
    "EL": "Greece",     "ES": "Spain",     "FI": "Finland",   "FR": "France",
    "HR": "Croatia",    "HU": "Hungary",   "IE": "Ireland",   "IT": "Italy",
    "LT": "Lithuania",  "LU": "Luxembourg","LV": "Latvia",    "MT": "Malta",
    "NL": "Netherlands","PL": "Poland",    "PT": "Portugal",  "RO": "Romania",
    "SE": "Sweden",     "SI": "Slovenia",  "SK": "Slovakia",
}
# Map Eurostat country codes to ISO 3166-1 alpha-3 (for Plotly choropleth)
CC_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE",
    "DE": "DEU", "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN", "IE": "IRL",
    "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT",
    "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE",
    "SI": "SVN", "SK": "SVK",
}


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame | None, dict | None,
                         pd.DataFrame | None, pd.DataFrame | None]:
    df = pd.read_parquet(GOLD_PATH)
    df["archetype_short"] = df["archetype_label"].map(ARCHETYPE_SHORT)
    df["country_name"]    = df["country_code"].map(COUNTRY_NAMES)
    df["iso3"]            = df["country_code"].map(CC_TO_ISO3)
    eff      = pd.read_csv(P6_EFF)       if P6_EFF.exists()       else None
    cospec   = pd.read_csv(P6D_PATTERNS) if P6D_PATTERNS.exists() else None
    cs_sum   = pd.read_csv(P6D_SUMMARY, index_col=0) if P6D_SUMMARY.exists() else None
    geojson  = json.loads(GEOJSON_PATH.read_text(encoding="utf-8")) if GEOJSON_PATH.exists() else None
    return df, eff, geojson, cospec, cs_sum


# ── Chart helpers ─────────────────────────────────────────────────────────────

def radar_chart(row: pd.Series, archetype_means: dict[str, list[float]]) -> go.Figure:
    categories = DIM_LABELS + [DIM_LABELS[0]]
    angles = np.linspace(0, 2 * np.pi, len(DIM_LABELS), endpoint=False)

    fig = go.Figure()

    # Archetype means (background reference)
    for label, means in archetype_means.items():
        vals = means + [means[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories,
            fill="toself", opacity=0.15,
            line=dict(color=ARCHETYPE_COLORS.get(label, "grey"), width=1, dash="dot"),
            name=ARCHETYPE_SHORT.get(label, label),
        ))

    # This region
    region_vals = [row[c] for c in MM_COLS] + [row[MM_COLS[0]]]
    fig.add_trace(go.Scatterpolar(
        r=region_vals, theta=categories,
        fill="toself", opacity=0.5,
        line=dict(color="#e07b39", width=2.5),
        name=str(row.name),
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0])),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=40, r=40, t=40, b=60),
        height=380,
    )
    return fig


def cohens_d_forest(eff: pd.DataFrame) -> go.Figure:
    eff_s = eff.sort_values("cohens_d")
    colors = [
        "#4c72b0" if d > 0.8 else
        "#55a868" if d > 0 else
        "#c44e52"
        for d in eff_s["cohens_d"]
    ]
    fig = go.Figure()
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.add_vline(x=0.8,  line_width=0.8, line_dash="dot", line_color="grey")
    fig.add_vline(x=-0.8, line_width=0.8, line_dash="dot", line_color="grey")

    for _, row in eff_s.iterrows():
        lbl = FEATURE_LABELS.get(row["feature_code"], row["feature_code"])
        dim = DIMENSION_MAP.get(row["feature_code"], "")
        col = ("#4c72b0" if row["cohens_d"] > 0.8 else
               "#55a868" if row["cohens_d"] > 0 else "#c44e52")
        fig.add_trace(go.Scatter(
            x=[row["cohens_d"]],
            y=[lbl],
            mode="markers",
            marker=dict(size=10, color=col, symbol="circle"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[row["ci_hi"] - row["cohens_d"]],
                arrayminus=[row["cohens_d"] - row["ci_lo"]],
                color=col,
                thickness=1.5,
                width=5,
            ),
            name=dim,
            showlegend=False,
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                f"d = {row['cohens_d']:.3f} "
                f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]<br>"
                f"MWU p = {row['mwu_pval']:.2e}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis_title="Cohen's d  (A2 high-performance − A1 catching-up)",
        height=520,
        margin=dict(l=200, r=30, t=30, b=50),
        plot_bgcolor="white",
        xaxis=dict(zeroline=False, showgrid=True, gridcolor="#eee"),
    )
    return fig


def country_choropleth(df: pd.DataFrame, metric: str, label: str) -> go.Figure:
    ctry = df.groupby(["country_code", "iso3", "country_name"])[metric].mean().reset_index()
    fig = px.choropleth(
        ctry,
        locations="iso3",
        color=metric,
        hover_name="country_name",
        hover_data={metric: ":.2f", "iso3": False},
        color_continuous_scale=[[0,"#0d1529"],[0.3,"#1a3a6b"],
                                 [0.6,"#0068cc"],[0.8,"#00a8ff"],[1,"#66d9e8"]],
        scope="europe",
        labels={metric: label},
    )
    fig.update_geos(
        scope="europe",
        bgcolor=_BG0, lakecolor=_BG0, landcolor="#0d1a2e",
        showcoastlines=True, coastlinecolor=_BORDER2, showframe=False,
        showocean=True, oceancolor=_BG0,
        showcountries=True, countrycolor=_BORDER2,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=420,
        coloraxis_colorbar=dict(title=label, thickness=10, len=0.65),
    )
    return fig


def nuts2_choropleth(
    df: pd.DataFrame,
    geojson: dict,
    metric: str,
    label: str,
    color_scale: str = "RdYlGn",
) -> go.Figure:
    """NUTS2-resolution choropleth using Eurostat GeoJSON."""
    plot_df = df.reset_index()[["nuts2_code", "nuts2_name", "country_code",
                                 "archetype_label", metric]].copy()
    fig = px.choropleth(
        plot_df,
        geojson=geojson,
        locations="nuts2_code",
        featureidkey="properties.NUTS_ID",
        color=metric,
        hover_name="nuts2_name",
        hover_data={
            "nuts2_code": True,
            "country_code": True,
            "archetype_label": True,
            metric: ":.3f",
        },
        color_continuous_scale=color_scale,
        labels={metric: label, "archetype_label": "Archetype"},
    )
    fig.update_geos(
        fitbounds="locations", visible=False,
        showcoastlines=True, coastlinecolor=_BORDER2,
        showland=True, landcolor="#0d1a2e",
        showframe=False, bgcolor=_BG0,
        showocean=True, oceancolor=_BG0,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=480,
        coloraxis_colorbar=dict(title=label, thickness=10, len=0.65),
    )
    return fig


def nuts2_archetype_map(df: pd.DataFrame, geojson: dict) -> go.Figure:
    """Discrete NUTS2 map coloured by archetype membership."""
    plot_df = df.reset_index()[["nuts2_code", "nuts2_name", "country_code",
                                 "archetype_label", "archetype_id",
                                 "silhouette_sample", "distance_to_centroid"]].copy()
    fig = px.choropleth(
        plot_df,
        geojson=geojson,
        locations="nuts2_code",
        featureidkey="properties.NUTS_ID",
        color="archetype_label",
        color_discrete_map=ARCHETYPE_COLORS,
        hover_name="nuts2_name",
        hover_data={
            "nuts2_code": True,
            "country_code": True,
            "silhouette_sample": ":.3f",
            "distance_to_centroid": ":.3f",
            "archetype_id": False,
        },
        labels={"archetype_label": "Archetype"},
        category_orders={"archetype_label": list(ARCHETYPE_COLORS.keys())},
    )
    fig.update_geos(
        fitbounds="locations", visible=False,
        showcoastlines=True, coastlinecolor=_BORDER2,
        showland=True, landcolor="#0d1a2e",
        showframe=False, bgcolor=_BG0,
        showocean=True, oceancolor=_BG0,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=500,
        legend=dict(orientation="h", y=-0.05, title="",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


def archetype_composition_bar(df: pd.DataFrame) -> go.Figure:
    counts = (df.groupby(["country_code", "archetype_label"])
              .size().unstack(fill_value=0))
    fracs  = counts.div(counts.sum(axis=1), axis=0)

    # Sort by A2 share
    a2_col = "Advanced innovation regions"
    if a2_col in fracs.columns:
        fracs = fracs.sort_values(a2_col)
    else:
        fracs = fracs.sort_index()

    fig = go.Figure()
    for col in fracs.columns:
        fig.add_trace(go.Bar(
            x=fracs.index.tolist(),
            y=fracs[col].values,
            name=ARCHETYPE_SHORT.get(col, col),
            marker_color=ARCHETYPE_COLORS.get(col, "grey"),
        ))
    fig.update_layout(
        barmode="stack",
        xaxis_title="Country",
        yaxis_title="Proportion of NUTS2 regions",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", y=-0.25),
        height=380,
        margin=dict(l=40, r=20, t=20, b=80),
    )
    return fig


def score_scatter(df: pd.DataFrame, xcol: str, ycol: str) -> go.Figure:
    fig = px.scatter(
        df.reset_index(),
        x=xcol, y=ycol,
        color="archetype_label",
        color_discrete_map=ARCHETYPE_COLORS,
        hover_data={"nuts2_code": True, "nuts2_name": True,
                    "country_code": True, xcol: ":.3f", ycol: ":.3f"},
        hover_name="nuts2_name",
        labels={xcol: xcol.replace("score_", "").replace("_", " ").title(),
                ycol: ycol.replace("score_", "").replace("_", " ").title(),
                "archetype_label": "Archetype"},
        opacity=0.7,
    )
    fig.add_hline(y=0, line_width=0.6, line_color="lightgrey")
    fig.add_vline(x=0, line_width=0.6, line_color="lightgrey")
    fig.update_layout(
        height=420, legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=20, b=80),
        plot_bgcolor="white",
    )
    return fig


def score_violin(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    fig = go.Figure()
    for label, color in ARCHETYPE_COLORS.items():
        vals = df.loc[df["archetype_label"] == label, col].dropna()
        fig.add_trace(go.Violin(
            y=vals, name=ARCHETYPE_SHORT.get(label, label),
            box_visible=True, meanline_visible=True,
            fillcolor=color, opacity=0.65,
            line_color=_BORDER2, line_width=0.8,
        ))
    fig.update_layout(
        yaxis_title=title, height=350,
        margin=dict(l=40, r=20, t=30, b=40),
        violinmode="group", showlegend=False,
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="EU Regional Innovation Panel",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Terminal-style header banner ──────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background:{_BG0};
            border:1px solid {_BORDER1};
            border-left:3px solid {_BLUE};
            padding:0.75rem 1.25rem;
            margin-bottom:0.75rem;
            border-radius:2px;
            display:flex;
            align-items:center;
            justify-content:space-between;
        ">
            <div>
                <span style="font-family:'JetBrains Mono','Courier New',monospace;
                             font-size:0.95rem;font-weight:500;
                             color:{_BLUE};letter-spacing:0.12em;">
                    EU REGIONAL INNOVATION PANEL
                </span>
                <span style="font-family:'JetBrains Mono','Courier New',monospace;
                             font-size:0.78rem;color:{_TEXT3};
                             margin-left:1.5rem;letter-spacing:0.06em;">
                    NUTS 2021 VINTAGE &nbsp;▪&nbsp; 242 REGIONS &nbsp;▪&nbsp; 27 MEMBER STATES
                </span>
            </div>
            <div style="display:flex;align-items:center;gap:1.2rem;">
                <span style="font-family:'JetBrains Mono','Courier New',monospace;
                             font-size:0.65rem;color:{_TEXT3};letter-spacing:0.1em;">
                    OBSERVATIONAL · NON-CAUSAL · DESCRIPTIVE
                </span>
                <span style="font-family:'JetBrains Mono','Courier New',monospace;
                             font-size:0.65rem;letter-spacing:0.1em;">
                    <span style="color:{_GREEN};">■</span>&nbsp;
                    <span style="color:{_TEXT2};">PIPELINE NOMINAL</span>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df, eff, geojson, cospec, cs_sum = load_data()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Region Explorer",
        "Score Analysis",
        "Barriers & Enablers",
        "Data Quality",
        "Co-Specialisation",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("Archetype summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total NUTS2 regions", len(df))
        c2.metric("Countries", df["country_code"].nunique())
        a1_n = int((df["archetype_id"] == 0).sum())
        a2_n = int((df["archetype_id"] == 1).sum())
        c3.metric("A1: Catching-up regions", a1_n)
        c4.metric("A2: Advanced innovation regions", a2_n)

        st.divider()
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown("**Archetype composition by country**")
            st.plotly_chart(
                archetype_composition_bar(df),
                use_container_width=True, key="comp_bar",
            )

        with col_right:
            if geojson is not None:
                st.markdown("**NUTS2 archetype map**")
                st.plotly_chart(
                    nuts2_archetype_map(df, geojson),
                    use_container_width=True, key="nuts2_map",
                )
            else:
                st.info("GeoJSON not found at data/raw/eurostat/nuts2_2021_geojson.json")

        if geojson is not None:
            st.divider()
            st.markdown("**NUTS2 metric choropleth**")
            metric_opts = {
                "GDP per capita (PPS)":         "gdp_per_capita_pps",
                "GERD % GDP":                   "rd_expenditure_pct_gdp",
                "HRST per 1 000 pop.":          "hrst_per_1000",
                "Score — Innovation cluster":   "score_cluster_minmax",
                "Score — Talent":               "score_talent_minmax",
                "Score — Prosperity":           "score_cost_minmax",
                "EPO patents / mio. pop.":      "epo_patents_per_mio_pop",
                "Enterprise internet use (%)":  "enterprise_internet_use",
            }
            map_choice = st.selectbox(
                "Metric", list(metric_opts.keys()), key="nuts2_metric_sel"
            )
            mkey = metric_opts[map_choice]
            st.plotly_chart(
                nuts2_choropleth(df, geojson, mkey, map_choice),
                use_container_width=True, key="nuts2_metric_map",
            )

        with st.expander("Country-level choropleth (aggregated)", expanded=False):
            cc_metric_opts = {
                "Mean GDP/capita (PPS)":      "gdp_per_capita_pps",
                "Mean GERD % GDP":            "rd_expenditure_pct_gdp",
                "Mean HRST per 1 000":        "hrst_per_1000",
                "Mean score (Innovation)":    "score_cluster_minmax",
                "Share A2 advanced regions":  "_a2_share",
            }
            cc_choice = st.selectbox(
                "Country metric", list(cc_metric_opts.keys()), key="cc_map_sel"
            )
            cc_key = cc_metric_opts[cc_choice]
            if cc_key == "_a2_share":
                tmp = df.copy()
                tmp["_a2_share"] = (tmp["archetype_id"] == 1).astype(float)
                fig_cc = country_choropleth(tmp, "_a2_share", "A2 share")
            else:
                fig_cc = country_choropleth(df, cc_key, cc_choice)
            st.plotly_chart(fig_cc, use_container_width=True, key="cc_choropleth")

        st.divider()
        st.markdown("**Archetype centroids (mean normalised scores)**")
        archetype_means: dict[str, list[float]] = {}
        for lbl in df["archetype_label"].unique():
            archetype_means[lbl] = df.loc[
                df["archetype_label"] == lbl, MM_COLS
            ].mean().tolist()

        centroid_df = pd.DataFrame(archetype_means, index=DIM_LABELS).T
        centroid_df.index = [ARCHETYPE_SHORT.get(i, i) for i in centroid_df.index]
        centroid_df.columns = DIM_LABELS
        st.dataframe(centroid_df.style.format("{:.3f}").background_gradient(
            cmap="RdYlGn", axis=None, vmin=0, vmax=1
        ), use_container_width=True)

    # ── Tab 2: Region Explorer ────────────────────────────────────────────────
    with tab2:
        st.subheader("Region profile explorer")
        st.caption(
            "Select a NUTS2 region to view its dimension scores, archetype membership, "
            "and raw feature values. Grey dotted lines show archetype centroid profiles "
            "as a descriptive reference."
        )

        # Build selector label: "DE21 — München (DE)"
        region_options = {
            f"{idx} — {row['nuts2_name']} ({row['country_code']})": idx
            for idx, row in df.iterrows()
        }
        sel_label = st.selectbox(
            "Select NUTS2 region", sorted(region_options.keys()), key="region_sel"
        )
        sel_code = region_options[sel_label]
        row = df.loc[sel_code]

        archetype_means_tab2: dict[str, list[float]] = {}
        for lbl in df["archetype_label"].unique():
            archetype_means_tab2[lbl] = df.loc[
                df["archetype_label"] == lbl, MM_COLS
            ].mean().tolist()

        c_left, c_right = st.columns([1, 1.3])

        with c_left:
            st.markdown(f"**{row['nuts2_name']}** &nbsp; `{sel_code}`")
            at_col = ARCHETYPE_COLORS.get(row["archetype_label"], "#888")
            st.markdown(
                f'<span style="background:{at_col};color:white;padding:3px 10px;'
                f'border-radius:4px;font-size:0.85rem">'
                f'{ARCHETYPE_SHORT.get(row["archetype_label"], row["archetype_label"])}'
                f"</span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                radar_chart(row, archetype_means_tab2),
                use_container_width=True, key="radar",
            )

        with c_right:
            st.markdown("**Dimension scores**")
            score_tbl = pd.DataFrame({
                "Dimension": DIM_LABELS,
                "z-score": [round(row[c], 3) for c in SCORE_COLS],
                "Normalised [0,1]": [round(row[c], 3) for c in MM_COLS],
            })
            st.dataframe(score_tbl, hide_index=True, use_container_width=True)

            st.markdown("**Quality flags**")
            flags = {
                "Transition region (GDP < 75 % EU27 mean)": bool(row["is_transition_region"]),
                "Insufficient data (quality score < 0.5)":  bool(row["is_insufficient_data"]),
                "Dimensionality warning (>3 imputed features)": bool(row["dimensionality_warning"]),
            }
            for flag_name, flag_val in flags.items():
                icon = "🔴" if flag_val else "🟢"
                st.write(f"{icon} {flag_name}")

            st.markdown("**Clustering metrics**")
            st.write(f"Silhouette coefficient: `{row['silhouette_sample']:.4f}`")
            st.write(f"Distance to centroid: `{row['distance_to_centroid']:.4f}`")
            st.write(f"Data quality score: `{row['data_quality_score']:.3f}`")

        st.divider()
        st.markdown("**Raw feature values vs archetype mean**")
        feat_data = []
        arch_mean = df[df["archetype_label"] == row["archetype_label"]][FEATURE_COLS].mean()
        eu27_mean = df[FEATURE_COLS].mean()
        for fc in FEATURE_COLS:
            v = row[fc]
            am = arch_mean[fc]
            em = eu27_mean[fc]
            imputed = bool(df.loc[sel_code, fc + "_imputed_flag"]) if fc + "_imputed_flag" in df.columns else False
            feat_data.append({
                "Feature":         FEATURE_LABELS[fc],
                "Dimension":       DIMENSION_MAP.get(fc, ""),
                "Value":           round(v, 3) if pd.notna(v) else "N/A",
                "Archetype mean":  round(am, 3),
                "EU27 mean":       round(em, 3),
                "vs arch. mean":   f"{'+' if v >= am else ''}{round(v - am, 3)}",
                "Imputed":         "Yes" if imputed else "",
            })
        st.dataframe(
            pd.DataFrame(feat_data),
            hide_index=True, use_container_width=True,
        )

    # ── Tab 3: Score Analysis ─────────────────────────────────────────────────
    with tab3:
        st.subheader("Dimension score distributions and pairwise associations")

        col_a, col_b = st.columns(2)
        with col_a:
            x_opt = st.selectbox("X axis", SCORE_COLS, index=0, key="sc_x")
        with col_b:
            y_opt = st.selectbox("Y axis", SCORE_COLS, index=3, key="sc_y")

        st.plotly_chart(
            score_scatter(df, x_opt, y_opt),
            use_container_width=True, key="scatter",
        )

        st.divider()
        st.markdown("**Score distributions by archetype (violin)**")
        v_cols = st.columns(4)
        for vcol, sc, lbl in zip(v_cols, SCORE_COLS, DIM_LABELS):
            with vcol:
                st.plotly_chart(
                    score_violin(df, sc, lbl),
                    use_container_width=True, key=f"violin_{sc}",
                )

    # ── Tab 4: Barriers & Enablers ────────────────────────────────────────────
    with tab4:
        st.subheader("Feature associations with archetype membership")
        st.caption(
            "Cohen's d measures the standardised mean difference between A2 (advanced innovation) "
            "and A1 (catching-up) archetypes. Positive d: feature is higher in A2. "
            "Negative d: feature is higher in A1 (catching-up regions). "
            "Error bars: 95 % bootstrap CI (n=999). * FDR-corrected (BH, alpha=0.05). "
            "Observational data — no causal claims are made."
        )

        if eff is not None:
            st.plotly_chart(
                cohens_d_forest(eff),
                use_container_width=True, key="forest",
            )

            st.divider()
            st.markdown("**Full effect size table**")
            eff_tbl = eff.copy()
            eff_tbl["Feature"] = eff_tbl["feature_code"].map(
                lambda x: FEATURE_LABELS.get(x, x)
            )
            eff_tbl["Dimension"] = eff_tbl["feature_code"].map(
                lambda x: DIMENSION_MAP.get(x, "")
            )
            eff_tbl["FDR sig."] = eff_tbl["fdr_reject"].map(
                lambda x: "Yes" if x else "No"
            )
            disp = eff_tbl[["Feature", "Dimension", "cohens_d",
                             "ci_lo", "ci_hi", "mwu_pval", "FDR sig."]].copy()
            disp.columns = ["Feature", "Dimension", "Cohen's d",
                            "95 % CI lo", "95 % CI hi", "MWU p-value", "FDR sig."]
            disp = disp.sort_values("Cohen's d", ascending=False)
            st.dataframe(
                disp.style.format({
                    "Cohen's d": "{:+.3f}",
                    "95 % CI lo": "{:.3f}",
                    "95 % CI hi": "{:.3f}",
                    "MWU p-value": "{:.2e}",
                }).background_gradient(
                    subset=["Cohen's d"],
                    cmap="RdYlGn", vmin=-2.5, vmax=2.5
                ),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("Effect size file not found. Run p6_barriers_enablers.py first.")

        st.divider()
        st.markdown("**Within-archetype feature heterogeneity** (coefficient of variation)")
        st.caption(
            "High CV within an archetype indicates that the two-group solution masks "
            "substantial internal variation for that feature."
        )
        cv_rows = []
        for fc in FEATURE_COLS:
            for aid, albl in [(0, "A1: Catching-up"), (1, "A2: Advanced")]:
                vals = df[df["archetype_id"] == aid][fc].dropna()
                mu = vals.mean()
                cv = vals.std(ddof=1) / abs(mu) if abs(mu) > 1e-9 else np.nan
                cv_rows.append({"Feature": FEATURE_LABELS[fc],
                                "Archetype": albl, "CV": round(cv, 3)})
        cv_df = pd.DataFrame(cv_rows).pivot(
            index="Feature", columns="Archetype", values="CV"
        )
        fig_cv = px.imshow(
            cv_df,
            color_continuous_scale="YlOrRd",
            labels=dict(color="CV"),
            aspect="auto",
            zmin=0,
            title="Coefficient of variation per feature per archetype",
        )
        fig_cv.update_layout(height=550, margin=dict(l=200, r=30, t=50, b=30))
        st.plotly_chart(fig_cv, use_container_width=True, key="cv_heat")

    # ── Tab 5: Data Quality ───────────────────────────────────────────────────
    with tab5:
        st.subheader("Data quality and coverage")

        q1, q2, q3 = st.columns(3)
        n_trans  = int(df["is_transition_region"].sum())
        n_insuf  = int(df["is_insufficient_data"].sum())
        n_dimwarn= int(df["dimensionality_warning"].sum())
        q1.metric("Transition regions (GDP < 75 % EU27)", n_trans)
        q2.metric("Insufficient data", n_insuf)
        q3.metric("Dimensionality warnings (>3 imputed features)", n_dimwarn)

        st.divider()
        st.markdown("**Imputation flags per feature**")
        imp_cols = [c for c in df.columns if c.endswith("_imputed_flag")]
        imp_summary = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(c.replace("_imputed_flag", ""), c)
                        for c in imp_cols],
            "Imputed count": [int(df[c].sum()) for c in imp_cols],
            "Imputed %":     [round(df[c].mean() * 100, 1) for c in imp_cols],
        }).sort_values("Imputed %", ascending=False)
        st.dataframe(imp_summary, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("**Data quality score distribution**")
        fig_dq = px.histogram(
            df,
            x="data_quality_score",
            nbins=20,
            color="archetype_label",
            color_discrete_map=ARCHETYPE_COLORS,
            barmode="overlay",
            opacity=0.75,
            labels={"data_quality_score": "Data quality score",
                    "archetype_label": "Archetype"},
            title="Distribution of data quality scores by archetype",
        )
        fig_dq.update_layout(height=350, margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig_dq, use_container_width=True, key="dq_hist")

        st.divider()
        st.markdown("**Country-level coverage: transition regions**")
        trans_by_cc = (
            df.groupby(["country_code", "country_name"])["is_transition_region"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "Transition", "count": "Total"})
        )
        trans_by_cc["Share (%)"] = (
            trans_by_cc["Transition"] / trans_by_cc["Total"] * 100
        ).round(1)
        trans_by_cc = trans_by_cc.sort_values("Share (%)", ascending=False)
        st.dataframe(
            trans_by_cc.reset_index().rename(
                columns={"country_code": "CC", "country_name": "Country"}
            ),
            hide_index=True, use_container_width=True,
        )


    # ── Tab 6: Co-Specialisation ──────────────────────────────────────────────
    with tab6:
        st.subheader("Sectoral co-specialisation typology")
        st.caption(
            "Each region is classified by which combination of sector location quotients "
            "exceeds LQ > 1.0 (over-represented relative to the EU average). "
            "LQ > 1.0 is a structural characteristic — it does not imply performance or "
            "investment attractiveness. Observational data, non-causal."
        )

        if cospec is None or cs_sum is None:
            st.info("Run scripts/p6d_cospec_typology.py to generate co-specialisation data.")
        else:
            PATTERN_ORDER_DASH = [
                "Diversified innovation", "Digital knowledge economy",
                "Digital-industrial", "Advanced mfg & R&D",
                "Energy-anchored mixed", "ICT services only",
                "KIS hi-tech only", "Hi-tech mfg only",
                "Energy/utilities only", "Non-specialised",
            ]
            PATTERN_COLORS_DASH = {
                "Diversified innovation":    "#2166ac",
                "Digital knowledge economy": "#4393c3",
                "Digital-industrial":        "#92c5de",
                "Advanced mfg & R&D":        "#74c476",
                "Energy-anchored mixed":     "#fd8d3c",
                "ICT services only":         "#abd9e9",
                "KIS hi-tech only":          "#a1d99b",
                "Hi-tech mfg only":          "#c7e9c0",
                "Energy/utilities only":     "#fdbe85",
                "Non-specialised":           "#d9d9d9",
            }

            # ── Summary metrics ───────────────────────────────────────────────
            n_cospec = int(cospec["cospec_pattern"].isin([
                "Diversified innovation", "Digital knowledge economy",
                "Digital-industrial", "Advanced mfg & R&D", "Energy-anchored mixed",
            ]).sum())
            n_spec = int((cospec["cospec_pattern"] != "Non-specialised").sum())
            n_div  = int((cospec["cospec_pattern"] == "Diversified innovation").sum())
            n_none = int((cospec["cospec_pattern"] == "Non-specialised").sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Regions with ≥ 1 specialisation", n_spec)
            m2.metric("Co-specialised (≥ 2 sectors)", n_cospec)
            m3.metric("Diversified innovation (3 sectors)", n_div)
            m4.metric("Non-specialised", n_none)

            st.divider()

            # ── Pattern × Archetype stacked bar ──────────────────────────────
            col_l, col_r = st.columns([1.3, 1])

            with col_l:
                st.markdown("**Pattern distribution by archetype**")
                pat_counts = (
                    cospec.groupby(["cospec_pattern", "archetype_label"])
                    .size().unstack(fill_value=0)
                )
                pat_counts = pat_counts.reindex(
                    [p for p in PATTERN_ORDER_DASH if p in pat_counts.index]
                )
                fig_pat = go.Figure()
                for arch, color in ARCHETYPE_COLORS.items():
                    if arch not in pat_counts.columns:
                        continue
                    fig_pat.add_trace(go.Bar(
                        y=pat_counts.index.tolist(),
                        x=pat_counts[arch].values,
                        orientation="h",
                        name=ARCHETYPE_SHORT.get(arch, arch),
                        marker_color=color,
                        text=pat_counts[arch].values,
                        textposition="inside",
                        insidetextanchor="middle",
                    ))
                fig_pat.update_layout(
                    barmode="stack",
                    xaxis_title="Number of NUTS2 regions",
                    height=420,
                    legend=dict(orientation="h", y=-0.15),
                    margin=dict(l=10, r=10, t=10, b=60),
                )
                st.plotly_chart(fig_pat, use_container_width=True, key="cospec_bar")

            with col_r:
                st.markdown("**Pattern summary table**")
                sum_display = cs_sum.copy()
                # Keep interpretable columns
                keep = [c for c in ["total", "a2_share", "pct_transition",
                                    "gdp_per_capita_pps", "hrst_per_1000",
                                    "rd_expenditure_pct_gdp"]
                        if c in sum_display.columns]
                sum_display = sum_display[keep].reindex(
                    [p for p in PATTERN_ORDER_DASH if p in sum_display.index]
                )
                rename = {
                    "total": "n",
                    "a2_share": "A2 share",
                    "pct_transition": "Trans. share",
                    "gdp_per_capita_pps": "Mean GDP/cap",
                    "hrst_per_1000": "Mean HRST/1k",
                    "rd_expenditure_pct_gdp": "Mean GERD %",
                }
                sum_display = sum_display.rename(columns=rename)
                fmt = {
                    "A2 share": "{:.0%}",
                    "Trans. share": "{:.0%}",
                    "Mean GDP/cap": "{:,.0f}",
                    "Mean HRST/1k": "{:.0f}",
                    "Mean GERD %": "{:.2f}",
                }
                st.dataframe(
                    sum_display.style.format(fmt, na_rep="—")
                    .background_gradient(subset=["A2 share"], cmap="RdYlGn",
                                         vmin=0, vmax=1),
                    use_container_width=True,
                )

            st.divider()

            # ── NUTS2 map coloured by pattern ─────────────────────────────────
            if geojson is not None:
                st.markdown("**Geographic distribution of co-specialisation patterns**")
                map_df = cospec[["nuts2_code", "cospec_pattern", "nuts2_name",
                                  "country_code"]].copy()
                fig_cs_map = px.choropleth(
                    map_df,
                    geojson=geojson,
                    locations="nuts2_code",
                    featureidkey="properties.NUTS_ID",
                    color="cospec_pattern",
                    color_discrete_map=PATTERN_COLORS_DASH,
                    hover_name="nuts2_name",
                    hover_data={"nuts2_code": True, "country_code": True,
                                "cospec_pattern": True},
                    labels={"cospec_pattern": "Pattern"},
                    category_orders={"cospec_pattern": PATTERN_ORDER_DASH},
                )
                fig_cs_map.update_geos(
                    fitbounds="locations", visible=False,
                    showcoastlines=True, coastlinecolor="lightgrey",
                    showland=True, landcolor="#f5f5f5", showframe=False,
                )
                fig_cs_map.update_layout(
                    height=520,
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", y=-0.08, title=""),
                )
                st.plotly_chart(fig_cs_map, use_container_width=True, key="cospec_map")

            st.divider()

            # ── LQ scatter for selected pattern ──────────────────────────────
            st.markdown("**LQ distributions by pattern**")
            LQ_COLS_DASH = [
                "lq_nace_j62j63", "lq_nace_c21_m72",
                "lq_nace_c26", "lq_nace_d35_clean",
            ]
            LQ_LABELS_DASH = {
                "lq_nace_j62j63":   "ICT services (J)",
                "lq_nace_c21_m72":  "KIS hi-tech (C21/M72)",
                "lq_nace_c26":      "Hi-tech mfg (C26)",
                "lq_nace_d35_clean": "Energy/utilities (D35)",
            }
            pat_filter = st.multiselect(
                "Filter patterns",
                options=PATTERN_ORDER_DASH,
                default=[p for p in PATTERN_ORDER_DASH
                         if p not in ("Non-specialised",)],
                key="cospec_pat_filter",
            )
            lq_df = cospec[cospec["cospec_pattern"].isin(pat_filter)] if pat_filter else cospec
            lq_long = lq_df.melt(
                id_vars=["nuts2_code", "nuts2_name", "cospec_pattern"],
                value_vars=LQ_COLS_DASH,
                var_name="sector", value_name="LQ",
            )
            lq_long["Sector"] = lq_long["sector"].map(LQ_LABELS_DASH)
            fig_lq = px.box(
                lq_long, x="Sector", y="LQ",
                color="cospec_pattern",
                color_discrete_map=PATTERN_COLORS_DASH,
                points=False,
                labels={"LQ": "Location quotient", "cospec_pattern": "Pattern"},
            )
            fig_lq.add_hline(y=1.0, line_dash="dot", line_color="black",
                             line_width=0.8, annotation_text="LQ = 1.0")
            fig_lq.update_layout(
                height=420,
                legend=dict(orientation="h", y=-0.25, title=""),
                margin=dict(l=40, r=20, t=20, b=100),
                yaxis=dict(range=[0, 4.0]),
            )
            st.plotly_chart(fig_lq, use_container_width=True, key="cospec_lq_box")


if __name__ == "__main__":
    main()
