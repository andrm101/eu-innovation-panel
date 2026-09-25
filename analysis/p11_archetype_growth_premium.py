"""Phase 11 -- Estimate A for Stage 13 Innovation ROI (consumed by
RO-Administrative-Reform and PL-Capital-Reform-DiD, cross-repo).

Regresses annualized regional GDP growth on archetype membership, controlling
for initial GDP level and population, to isolate an archetype-associated
growth premium. Non-causal: this is a cross-sectional regression on
observational data, reported per this repo's vocabulary contract as an
association, not a causal effect (see CLAUDE.md).

Data source: Eurostat nama_10r_2gdp (NUTS2 GDP per capita, PPS), following
this repo's existing SDMX-CSV ingestion pattern (see p2_ingest_harmonize.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf

ROOT = Path(__file__).parent.parent
EUROSTAT_SDMX_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
    "nama_10r_2gdp/?format=SDMX-CSV&lang=en"
)
# GDP per capita in PPS, EU27_2020=100 basket -- the SDMX response mixes several
# `unit` dimension values (MIO_EUR, MIO_NAC, EUR_HAB, ...) in one unfiltered pull;
# must select this one explicitly or nuts2_code+year keys are not unique.
GDP_UNIT = "PPS_HAB_EU27_2020"
YEAR_START = 2012
YEAR_END = 2023
SEED = 42


def fetch_gdp_panel() -> pd.DataFrame:
    """Fetch Eurostat nama_10r_2gdp, return long-form nuts2_code/year/gdp_pc."""
    resp = requests.get(EUROSTAT_SDMX_URL, timeout=120)
    resp.raise_for_status()
    raw = pd.read_csv(pd.io.common.StringIO(resp.text))
    raw = raw.rename(columns={"geo": "nuts2_code", "TIME_PERIOD": "year", "OBS_VALUE": "gdp_pc"})
    raw = raw[raw["unit"] == GDP_UNIT]
    raw = raw[raw["nuts2_code"].str.len() == 4]  # NUTS2 codes only
    out = raw[["nuts2_code", "year", "gdp_pc"]].dropna()
    dup_key = out.duplicated(subset=["nuts2_code", "year"], keep=False)
    if dup_key.any():
        raise ValueError(
            f"fetch_gdp_panel: {dup_key.sum()} duplicate (nuts2_code, year) rows "
            f"remain after filtering to unit={GDP_UNIT!r} -- Eurostat response has "
            f"an unexpected extra dimension; inspect raw['freq'] etc."
        )
    return out


def build_growth_panel(gdp_long: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Collapse long-form GDP panel to one row per region: start/end GDP,
    joined to archetype_id and population from the Gold layer.

    `gold` carries its region identifier as the index (named nuts2_code),
    not as a column -- confirmed against data/gold/region_profiles_gold.parquet.
    """
    start = gdp_long[gdp_long["year"] == YEAR_START].set_index("nuts2_code")["gdp_pc"]
    end = gdp_long[gdp_long["year"] == YEAR_END].set_index("nuts2_code")["gdp_pc"]
    both = pd.DataFrame({"gdp_pc_start": start, "gdp_pc_end": end}).dropna().reset_index()
    both["n_years"] = YEAR_END - YEAR_START
    gold_flat = gold[["archetype_id", "archetype_label", "population"]].reset_index()  # index -> nuts2_code column
    merged = both.merge(gold_flat, on="nuts2_code", how="inner")
    return merged


def fit_growth_premium(df: pd.DataFrame) -> pd.DataFrame:
    """df: nuts2_code, archetype_id, gdp_pc_start, gdp_pc_end, n_years, population.
    Returns one row per archetype: growth_premium_pp (vs. the omitted baseline
    archetype), se, ci_lo, ci_hi (95%), n_regions.
    """
    work = df.copy()
    work["annual_growth_pct"] = (
        (np.log(work["gdp_pc_end"] / work["gdp_pc_start"]) / work["n_years"]) * 100
    )
    work["ln_gdp_start"] = np.log(work["gdp_pc_start"])
    work["ln_population"] = np.log(work["population"])

    model = smf.ols(
        "annual_growth_pct ~ C(archetype_id) + ln_gdp_start + ln_population",
        data=work,
    ).fit()

    has_label = "archetype_label" in work.columns
    rows = []
    for archetype in sorted(work["archetype_id"].unique()):
        subset = work[work["archetype_id"] == archetype]
        n_regions = int(len(subset))
        term = f"C(archetype_id)[T.{archetype}]"
        if term in model.params.index:
            coef = model.params[term]
            se = model.bse[term]
            ci_lo, ci_hi = model.conf_int().loc[term]
        else:
            # Omitted (baseline) category: premium is 0 by construction
            coef, se, ci_lo, ci_hi = 0.0, 0.0, 0.0, 0.0
        row = {
            "archetype_id": archetype,
            "growth_premium_pp": coef,
            "se": se,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_regions": n_regions,
        }
        if has_label:
            row["archetype_label"] = subset["archetype_label"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(SEED)
    gold = pd.read_parquet(ROOT / "data" / "gold" / "region_profiles_gold.parquet")
    gdp_long = fetch_gdp_panel()
    panel = build_growth_panel(gdp_long, gold)
    out = fit_growth_premium(panel)
    out_path = ROOT / "analysis" / "p11_archetype_growth_premium.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
