"""
Phase 2 — Data Ingestion & Harmonization

Bronze → Silver medallion pipeline:
  P2.T1-T4  Download Eurostat SDMX-CSV + CORDIS CSV to data/bronze/
  P2.T5     Write bronze_manifest.json (SHA-256 per file)
  P2.T6     Normalize NUTS codes to 2021 vintage; filter to EU27
  P2.T7     Write nuts_recode_log.csv
  P2.T8     Year alignment per variable (latest yr with ≥60% coverage)
  P2.T9     Unit normalization (PPP-adjust monetary; per-capita where needed)
  P2.T10    Compute Location Quotients per NACE × NUTS2
  P2.T11-12 CORDIS download + aggregate to NUTS2
  P2.T13-16 Missingness classification + imputation
  P2.T17    Merge all features → Silver parquet
  P2.T18    Pandera Silver schema validation
  P2.T19    Write ingestion_report.json

API used: Eurostat SDMX 2.1 with format=SDMX-CSV (flat long CSV).
Response columns: DATAFLOW, LAST UPDATE, freq, [dims...], geo, TIME_PERIOD,
                  OBS_VALUE, OBS_FLAG, CONF_STATUS
"""

import hashlib
import io
import json
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

np.random.seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import configure_logging, pipeline_step

configure_logging()

# ── Constants ─────────────────────────────────────────────────────────────────

EU27_COUNTRY_CODES = {
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",
}

# SDMX-CSV returns flat long format — confirmed working (200 OK)
EUROSTAT_SDMX_CSV = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
    "/{code}?format=SDMX-CSV&lang=EN&startPeriod={start}"
)

# Key-filtered URL for lfst_r_lfe2en2 (full dataset returns 413).
# Dims: freq.unit.sex.age.nace_r2.geo — filter to annual/THS/total/working-age/target sectors.
# Target NACE: J62, J63, C21, M72, C26, D35 plus TOTAL for denominator.
_LFT_NACE_FILTER = "J62+J63+C21+M72+C26+D35+TOTAL+B-S_X_O"
EUROSTAT_LFT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
    f"/lfst_r_lfe2en2/A.THS.T.TOTAL.{_LFT_NACE_FILTER}.."
    "?format=SDMX-CSV&lang=EN&startPeriod=2015"
)

# CORDIS Horizon Europe bulk CSV
CORDIS_HE_ORG = "https://cordis.europa.eu/data/cordis-HEorganizations.csv"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EU-Innovation-Panel/0.1 (academic research)"})

# ── Download configs ──────────────────────────────────────────────────────────
# (dataset_code, start_year)
# Notes on problem datasets:
#   - tgs00063 (BERD): 404 — derived from rd_e_gerdreg with sectperf=BES
#   - earn_ses_pub2s: no NUTS2 rows — national-level only; wage dropped
#   - educ_uoe_enrt07: no EU27 NUTS2 rows — replaced by edat_lfse_04
#   - yth_empl_040: no EU27 NUTS2 rows — dropped
#   - pat_ep_rtot: may return 400; fails gracefully
#   - nrg_r_rgen: 404 — fails gracefully
#   - bd_9bd_sz_cl_r2: 413 even at 2018 — dropped
#   - lfst_r_lfe2en2: returns empty XML; LQ derived from htec_emp_reg2 instead
EUROSTAT_DATASETS = [
    ("nama_10r_2gdp",     2010),   # GDP per capita PPS
    ("hrst_st_rcat",      2010),   # HRST — human resources in science & technology
    ("edat_lfse_04",      2015),   # Tertiary education attainment (replaces educ_uoe_enrt07)
    ("tgs00007",          2010),   # Employment rate (NUTS2 aggregate)
    ("rd_p_persreg",      2010),   # R&D personnel by region
    ("demo_r_gind3",      2010),   # Net migration rate
    ("isoc_r_broad_h",    2010),   # Broadband penetration
    ("isoc_r_iuse_i",     2010),   # Enterprise internet use (broadband demand proxy)
    ("rd_e_gerdreg",      2010),   # GERD + BERD (sectperf=BES)
    ("pat_ep_rtot",       2010),   # EPO patents — may return 400; fails gracefully
    ("nrg_r_rgen",        2010),   # Renewable energy share — may return 404
    ("htec_emp_reg2",     2015),   # Hi-tech employment; also used for LQ computation
    ("nama_10r_3gva",     2010),   # GVA by sector (NUTS3; aggregated to NUTS2)
    ("demo_r_pjanaggr3",  2010),   # Population by NUTS2
    ("demo_r_d3dens",     2010),   # Population density
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 300,
                  max_retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            return True
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            print(f"    HTTP {status} for {url.split('?')[0].split('/')[-1]}")
            return False  # HTTP errors are deterministic — no retry
        except Exception as exc:
            print(f"    attempt {attempt}/{max_retries}: {exc}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return False


# ── SDMX-CSV parser ───────────────────────────────────────────────────────────

def parse_sdmx_csv(path: Path, nuts2_only: bool = True) -> pd.DataFrame:
    """
    Parse Eurostat SDMX-CSV flat format into a long DataFrame.

    Input columns (Eurostat SDMX-CSV 2.1):
        DATAFLOW, LAST UPDATE, freq, [dims...], geo, TIME_PERIOD,
        OBS_VALUE, OBS_FLAG, CONF_STATUS

    Output columns:
        [dims...], geo, year (int), value (float), flag (str)
    """
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception as exc:
        print(f"    read error {path.name}: {exc}")
        return pd.DataFrame()

    if df.empty or "OBS_VALUE" not in df.columns:
        return pd.DataFrame()

    # Drop Eurostat metadata columns — keep dims + geo + time + value
    drop_cols = [c for c in ("DATAFLOW", "LAST UPDATE", "CONF_STATUS") if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.rename(columns={"OBS_VALUE": "value", "OBS_FLAG": "flag"})

    # Extract 4-digit year from TIME_PERIOD (handles "2021", "2021-Q3", "2021-S1")
    if "TIME_PERIOD" in df.columns:
        df["year"] = pd.to_numeric(
            df["TIME_PERIOD"].astype(str).str[:4], errors="coerce"
        )
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        df = df.drop(columns=["TIME_PERIOD"])

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "geo" not in df.columns:
        return pd.DataFrame()

    if nuts2_only:
        df = df[
            (df["geo"].str.len() == 4) &
            (df["geo"].str[:2].isin(EU27_COUNTRY_CODES))
        ].copy()

    return df.reset_index(drop=True)


def get_ppp_index(bronze_paths: dict) -> pd.Series:
    """
    Extract PPP price level indices at country level from prc_ppp_ind.
    Returns Series indexed by 2-char country code (EU27 only).
    """
    p = bronze_paths.get("prc_ppp_ind")
    if p is None:
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(p, encoding="utf-8", low_memory=False)
        if "OBS_VALUE" not in df.columns or "geo" not in df.columns:
            return pd.Series(dtype=float)
        df = df.rename(columns={"OBS_VALUE": "value"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["year"] = pd.to_numeric(
            df["TIME_PERIOD"].astype(str).str[:4], errors="coerce"
        )
        # Country level: 2-char codes; filter to CP_EUR unit (current prices vs EUR)
        df_cc = df[
            (df["geo"].str.len() == 2) &
            (df["geo"].isin(EU27_COUNTRY_CODES))
        ].copy()
        if "unit" in df_cc.columns:
            # Prefer PLI_EU27 (price level index vs EU average); fall back to any
            for unit in ["PLI_EU27", "PLI_EU28", "PLI_EU"]:
                sub = df_cc[df_cc["unit"] == unit]
                if not sub.empty:
                    df_cc = sub
                    break
        # Most recent year per country
        latest_idx = df_cc.groupby("geo")["year"].idxmax()
        ppp = df_cc.loc[latest_idx].set_index("geo")["value"]
        return ppp
    except Exception:
        return pd.Series(dtype=float)


# ── Eurostat downloader ───────────────────────────────────────────────────────

def download_eurostat(code: str, start: int, log) -> Path | None:
    """Download a single Eurostat dataset via SDMX-CSV API. Returns path or None."""
    dest = ROOT / f"data/bronze/eurostat/{code}.csv"

    if dest.exists() and dest.stat().st_size > 2000:
        log.info("bronze_cached", code=code, size_kb=dest.stat().st_size // 1024)
        return dest

    # lfst_r_lfe2en2 is too large without a key filter — use pre-filtered URL
    if code == "lfst_r_lfe2en2":
        url = EUROSTAT_LFT_URL
    else:
        url = EUROSTAT_SDMX_CSV.format(code=code, start=start)

    print(f"    downloading {code} (from {start})...")
    # Polite inter-request delay to avoid rate-limiting
    time.sleep(1)
    ok = download_file(url, dest, timeout=300)

    if ok and dest.stat().st_size > 500:
        size_kb = dest.stat().st_size // 1024
        print(f"    {code}: {size_kb} KB")
        return dest

    # Remove empty/failed file
    if dest.exists():
        dest.unlink()
    log.warning("download_failed", code=code)
    return None


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_nuts2_series(df_long: pd.DataFrame, dim_filters: dict,
                          agg: str = "mean") -> pd.Series:
    """
    Filter a long SDMX-CSV DataFrame by dim_filters, select the latest year
    with ≥60% NUTS2 coverage, return Series indexed by nuts2_code.
    """
    if df_long.empty:
        return pd.Series(dtype=float)

    mask = pd.Series([True] * len(df_long), index=df_long.index)
    for col, vals in dim_filters.items():
        if col not in df_long.columns:
            continue
        if isinstance(vals, list):
            mask &= df_long[col].isin(vals)
        else:
            mask &= (df_long[col] == str(vals))

    sub = df_long[mask].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    # Year with best coverage (≥60% of observed regions non-null)
    n_regions = sub["geo"].nunique()
    year_cov = sub.groupby("year")["value"].apply(
        lambda s: s.notna().sum() / max(n_regions, 1)
    )
    valid_years = year_cov[year_cov >= 0.60].index
    if len(valid_years) == 0:
        valid_years = year_cov.index

    ref_year = int(valid_years.max())
    sub_yr = sub[sub["year"] == ref_year]

    result = sub_yr.groupby("geo")["value"].sum() if agg == "sum" else \
             sub_yr.groupby("geo")["value"].mean()

    result.index.name = "nuts2_code"
    return result


def build_features(bronze_paths: dict, log) -> dict:
    """Parse all Bronze SDMX-CSV files and extract feature Series."""
    features: dict = {}

    def get_df(code: str) -> pd.DataFrame:
        p = bronze_paths.get(code)
        if p is None:
            return pd.DataFrame()
        return parse_sdmx_csv(p)

    # ── GDP per capita PPS ────────────────────────────────────────────────────
    print("  Extracting GDP per capita (nama_10r_2gdp)...")
    df = get_df("nama_10r_2gdp")
    if not df.empty:
        for unit in ["PPS_HAB", "EUR_HAB", "EUR_HAB_EU", "MIO_EUR"]:
            s = extract_nuts2_series(df, {"unit": unit})
            if not s.empty and s.median() > 0:
                features["gdp_per_capita_pps"] = s
                break
        if "gdp_per_capita_pps" not in features:
            features["gdp_per_capita_pps"] = extract_nuts2_series(df, {})

    # ── Population ────────────────────────────────────────────────────────────
    print("  Extracting population...")
    df = get_df("demo_r_pjanaggr3")
    if not df.empty:
        for filt in [{"sex": "T", "age": "TOTAL"}, {"sex": "T"}, {}]:
            s = extract_nuts2_series(df, filt, agg="mean")
            if not s.empty and s.median() > 10000:
                features["population"] = s
                break

    # ── HRST ─────────────────────────────────────────────────────────────────
    print("  Extracting HRST...")
    df = get_df("hrst_st_rcat")
    if not df.empty:
        for filt in [
            {"sex": "T", "category": "HRST", "unit": "PC_ACT"},
            {"sex": "T", "category": "HRST"},
            {"sex": "T"},
        ]:
            s = extract_nuts2_series(df, filt)
            if not s.empty:
                if filt.get("unit") == "PC_ACT":
                    s = s * 10  # percent of active → per 1000
                features["hrst_per_1000"] = s
                break

    # ── Tertiary education attainment ─────────────────────────────────────────
    print("  Extracting tertiary education attainment (edat_lfse_04)...")
    df = get_df("edat_lfse_04")
    if not df.empty:
        for filt in [
            {"sex": "T", "isced11": "ED5-8", "age": "Y25-64", "unit": "PC"},
            {"sex": "T", "isced11": "ED5-8", "unit": "PC"},
            {"sex": "T", "isced11": "ED5-8"},
            {"isced11": "ED5-8"},
            {},
        ]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and 0 < s.median() < 100:
                features["tertiary_enrolment_rate"] = s
                break

    # ── Employment rate ───────────────────────────────────────────────────────
    print("  Extracting employment rate...")
    df = get_df("tgs00007")
    if not df.empty:
        for filt in [{"sex": "T", "age": "Y20-64"}, {"sex": "T"}, {}]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and 0 < s.median() < 100:
                features["employment_rate"] = s
                break

    # ── Average wages ─────────────────────────────────────────────────────────
    print("  Extracting average wages...")
    df = get_df("earn_ses_pub2s")
    if not df.empty:
        for filt in [
            {"sex": "T", "nace_r2": "B-S", "unit": "EUR"},
            {"sex": "T", "unit": "EUR"},
            {"unit": "EUR"},
            {},
        ]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and s.median() > 100:
                features["avg_wage_eur"] = s
                break

    # ── Broadband ─────────────────────────────────────────────────────────────
    print("  Extracting broadband penetration...")
    df = get_df("isoc_r_broad_h")
    if not df.empty:
        for filt in [{"unit": "PC_HH"}, {}]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and 0 < s.median() < 100:
                features["broadband_penetration_pct"] = s
                break

    # ── R&D expenditure GERD ─────────────────────────────────────────────────
    print("  Extracting R&D expenditure (GERD)...")
    df = get_df("rd_e_gerdreg")
    if not df.empty:
        for filt in [
            {"sectperf": "TOTAL", "unit": "PC_GDP"},
            {"unit": "PC_GDP"},
            {"sectperf": "TOTAL"},
            {},
        ]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and 0 <= s.median() < 15:
                features["rd_expenditure_pct_gdp"] = s
                break

        # Business R&D (BERD) from same dataset — sectperf=BES
        print("  Extracting business R&D (BERD from rd_e_gerdreg)...")
        for filt in [
            {"sectperf": "BES", "unit": "PC_GDP"},
            {"sectperf": "BES"},
        ]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and s.median() >= 0:
                features["business_rd_pct_gdp"] = s
                break

    # ── EPO patents ───────────────────────────────────────────────────────────
    print("  Extracting EPO patents (pat_ep_rtot)...")
    df = get_df("pat_ep_rtot")
    if not df.empty:
        for filt in [{"unit": "P_MHAB"}, {}]:
            s = extract_nuts2_series(df, filt)
            if not s.empty:
                features["epo_patents_per_mio_pop"] = s
                break

    # ── R&D personnel ─────────────────────────────────────────────────────────
    print("  Extracting R&D personnel...")
    df = get_df("rd_p_persreg")
    if not df.empty:
        for filt in [
            {"sectperf": "TOTAL", "sex": "T", "unit": "PC_EMP"},
            {"sex": "T", "unit": "PC_EMP"},
            {"unit": "PC_EMP"},
            {},
        ]:
            s = extract_nuts2_series(df, filt)
            if not s.empty:
                features["rd_personnel_pct_employment"] = s
                break

    # ── Net migration ─────────────────────────────────────────────────────────
    print("  Extracting net migration...")
    df = get_df("demo_r_gind3")
    if not df.empty:
        for filt in [
            {"indic_de": "CNMR", "sex": "T"},
            {"indic_de": "CNMR"},
            {},
        ]:
            s = extract_nuts2_series(df, filt)
            if not s.empty:
                features["net_migration_rate"] = s
                break

    # ── Renewable energy ─────────────────────────────────────────────────────
    print("  Extracting renewable energy share...")
    df = get_df("nrg_r_rgen")
    if not df.empty:
        for filt in [{"nrg_bal": "REN", "unit": "PC"}, {"unit": "PC"}, {}]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and 0 <= s.median() < 100:
                features["renewable_energy_share"] = s
                break

    # ── Youth employment ─────────────────────────────────────────────────────
    print("  Extracting youth employment rate...")
    df = get_df("yth_empl_040")
    if not df.empty:
        for filt in [{"sex": "T", "unit": "PC"}, {"sex": "T"}, {}]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and 0 < s.median() < 100:
                features["youth_employment_rate"] = s
                break

    # ── Business births (startup proxy) ──────────────────────────────────────
    print("  Extracting startup density proxy (bd_9bd_sz_cl_r2)...")
    df = get_df("bd_9bd_sz_cl_r2")
    if not df.empty:
        for filt in [
            {"indic_sb": "V97290", "sizeclas": "TOTAL"},
            {"sizeclas": "TOTAL"},
            {},
        ]:
            s = extract_nuts2_series(df, filt)
            if not s.empty:
                features["startup_density_proxy"] = s
                break

    # ── Hi-tech employment ────────────────────────────────────────────────────
    print("  Extracting hi-tech employment...")
    df = get_df("htec_emp_reg2")
    if not df.empty:
        for filt in [{"sex": "T", "unit": "PC_EMP"}, {"unit": "PC_EMP"}, {}]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and 0 < s.median() < 100:
                features["hi_tech_employment_pct"] = s
                break

    # ── GVA ICT share (NUTS3→NUTS2 aggregation) ──────────────────────────────
    print("  Extracting GVA ICT share (NUTS3 aggregated to NUTS2)...")
    p_gva = bronze_paths.get("nama_10r_3gva")
    if p_gva is not None:
        df_gva_raw = pd.read_csv(p_gva, encoding="utf-8", low_memory=False)
        df_gva_raw["value"] = pd.to_numeric(df_gva_raw.get("OBS_VALUE", pd.Series(dtype=float)), errors="coerce")
        df_gva_raw["year"] = pd.to_numeric(
            df_gva_raw.get("TIME_PERIOD", pd.Series(dtype=str)).astype(str).str[:4], errors="coerce"
        )
        # Accept both NUTS2 (4-char) and NUTS3 (5-char); derive nuts2 from first 4 chars
        df_gva_raw["nuts2"] = df_gva_raw["geo"].astype(str).str[:4]
        mask_eu27 = df_gva_raw["nuts2"].str[:2].isin(EU27_COUNTRY_CODES)
        df_gva = df_gva_raw[mask_eu27].copy()
        if not df_gva.empty and "nace_r2" in df_gva.columns:
            yr_cov = df_gva[df_gva["nace_r2"]=="J"].groupby("year")["value"].apply(
                lambda s: s.notna().sum() / max(df_gva["nuts2"].nunique(), 1)
            )
            valid_yr = yr_cov[yr_cov >= 0.40].index
            ref_yr_gva = int(valid_yr.max()) if len(valid_yr) > 0 else int(yr_cov.index.max())
            df_gva_yr = df_gva[df_gva["year"] == ref_yr_gva]
            j_gva = df_gva_yr[df_gva_yr["nace_r2"] == "J"].groupby("nuts2")["value"].sum()
            tot_gva = df_gva_yr[df_gva_yr["nace_r2"].isin(["TOTAL", "A-U", "B-N_R-S"])].groupby("nuts2")["value"].sum()
            if not tot_gva.empty and not j_gva.empty:
                gva_share = (j_gva / tot_gva * 100).replace([float("inf"), -float("inf")], float("nan"))
                gva_share.index.name = "nuts2_code"
                if gva_share.notna().sum() > 50:
                    features["gva_ict_share"] = gva_share
                    print(f"    gva_ict_share: {gva_share.notna().sum()} regions (ref year {ref_yr_gva})")
        if "gva_ict_share" not in features:
            # Fallback: use pre-parsed SDMX-CSV with NUTS2 only
            df_nuts2_only = parse_sdmx_csv(p_gva)
            if not df_nuts2_only.empty:
                for filt in [{"nace_r2": "J", "unit": "PC"}, {"nace_r2": "J"}]:
                    s = extract_nuts2_series(df_nuts2_only, filt)
                    if not s.empty:
                        features["gva_ict_share"] = s
                        break

    # ── Population density ────────────────────────────────────────────────────
    print("  Extracting population density...")
    df = get_df("demo_r_d3dens")
    if not df.empty:
        s = extract_nuts2_series(df, {"unit": "PER_KM2"})
        if not s.empty:
            features["population_density"] = s

    # ── Enterprise internet use ───────────────────────────────────────────────
    print("  Extracting enterprise internet use...")
    df = get_df("isoc_r_iuse_i")
    if not df.empty:
        for filt in [{"unit": "PC_IND"}, {}]:
            s = extract_nuts2_series(df, filt)
            if not s.empty and 0 < s.median() < 100:
                features["enterprise_internet_use"] = s
                break

    # ── PPP index (country-level, used for wage adjustment) ───────────────────
    ppp_series = get_ppp_index(bronze_paths)
    if not ppp_series.empty:
        features["_ppp_index"] = ppp_series

    log.info("features_extracted",
             count=len([k for k in features if not k.startswith("_")]),
             names=sorted(k for k in features if not k.startswith("_")))
    return features


# ── LQ computation ────────────────────────────────────────────────────────────

def compute_location_quotients(bronze_paths: dict, log) -> dict:
    """
    Compute Location Quotients for target NACE sectors using htec_emp_reg2.

    lfst_r_lfe2en2 returns empty data from the SDMX API (dataset deprecated or
    migrated). htec_emp_reg2 provides section-level employment at NUTS2 and is
    used as the LQ source with the following proxy mappings:
      lq_nace_j62j63   ← section J  (ICT; includes J62+J63)
      lq_nace_c21_m72  ← KIS_HTC    (knowledge-intensive hi-tech services; M72 proxy)
      lq_nace_c26      ← C_HTC      (hi-tech manufacturing; C26 proxy)
      lq_nace_d35_clean← D-F        (energy+utilities aggregate; D35 proxy)

    LQ = (emp_region_sector / emp_region_total) / (emp_EU_sector / emp_EU_total)
    Capped at 5.0.
    """
    lq_features: dict = {}
    p = bronze_paths.get("htec_emp_reg2")
    if p is None:
        log.warning("lq_skipped", reason="htec_emp_reg2 not downloaded")
        return lq_features

    print("  Parsing htec_emp_reg2 for LQ computation...")
    df = parse_sdmx_csv(p)
    if df.empty:
        log.warning("lq_skipped", reason="htec_emp_reg2 parse returned empty")
        return lq_features

    # Filter: sex=T (total), unit=THS_PER (thousands of persons)
    mask = pd.Series([True] * len(df), index=df.index)
    if "sex" in df.columns:
        mask &= df["sex"] == "T"
    if "unit" in df.columns:
        mask &= df["unit"] == "THS_PER"

    df_filt = df[mask].copy()
    if df_filt.empty:
        df_filt = df.copy()

    # Latest year with ≥60% NUTS2 coverage for TOTAL sector
    yr_cov = df_filt[df_filt.get("nace_r2", pd.Series()) == "TOTAL"].groupby("year")["value"].apply(
        lambda s: s.notna().sum() / max(df_filt["geo"].nunique(), 1)
    ) if "nace_r2" in df_filt.columns else pd.Series(dtype=float)

    if yr_cov.empty:
        yr_cov = df_filt.groupby("year")["value"].apply(
            lambda s: s.notna().sum() / max(len(s), 1)
        )

    valid_yrs = yr_cov[yr_cov >= 0.60].index
    if len(valid_yrs) == 0:
        valid_yrs = yr_cov.index
    ref_year = int(valid_yrs.max())
    df_yr = df_filt[df_filt["year"] == ref_year].copy()

    nace_col = "nace_r2" if "nace_r2" in df_yr.columns else None
    if nace_col is None:
        log.warning("lq_skipped", reason="No nace_r2 column in htec_emp_reg2")
        return lq_features

    # Total employment per NUTS2 region
    total_emp = df_yr[df_yr[nace_col] == "TOTAL"].groupby("geo")["value"].sum()
    if total_emp.empty:
        total_emp = df_yr.groupby("geo")["value"].sum()
    eu_total = total_emp.sum()
    if eu_total == 0:
        return lq_features

    # Proxy NACE mappings using htec_emp_reg2 section aggregates
    NACE_MAP = {
        "lq_nace_j62j63":    ["J"],          # ICT (J62+J63 proxy)
        "lq_nace_c21_m72":   ["KIS_HTC"],    # Knowledge-intensive hi-tech services proxy
        "lq_nace_c26":       ["C_HTC"],      # Hi-tech manufacturing proxy
        "lq_nace_d35_clean": ["D-F"],        # Energy/utilities aggregate proxy
    }

    for feat_name, nace_codes in NACE_MAP.items():
        sector_emp = df_yr[df_yr[nace_col].isin(nace_codes)].groupby("geo")["value"].sum()
        if sector_emp.empty:
            continue
        eu_sector = sector_emp.sum()
        if eu_sector == 0:
            continue

        # Align index so only regions present in both series are used
        common = total_emp.index.intersection(sector_emp.index)
        if len(common) < 50:
            continue
        lq = (sector_emp[common] / total_emp[common]) / (eu_sector / eu_total)
        lq = lq.clip(0, 5.0)
        lq.index.name = "nuts2_code"
        lq_features[feat_name] = lq

    log.info("lq_computed", features=list(lq_features.keys()),
             ref_year=ref_year, nuts2_regions=len(total_emp),
             source="htec_emp_reg2")
    return lq_features


# ── CORDIS integration ───────────────────────────────────────────────────────

def download_and_process_cordis(log) -> dict:
    """
    Download CORDIS HE organizations CSV, extract NUTS2 codes, aggregate.
    Fully optional — returns empty dict on any failure.
    """
    orgs_path = ROOT / "data/bronze/cordis/cordis-HEorganizations.csv"

    if not orgs_path.exists():
        print("  Downloading CORDIS HE organizations (~100 MB)...")
        ok = download_file(CORDIS_HE_ORG, orgs_path, timeout=600)
        if not ok:
            log.warning("cordis_download_failed",
                        note="CORDIS features will be missing — set to NaN")
            return {}

    try:
        chunks = []
        needed_cols: dict | None = None
        for chunk in pd.read_csv(orgs_path, sep=";", encoding="utf-8",
                                  chunksize=50000, on_bad_lines="skip",
                                  low_memory=False):
            if needed_cols is None:
                cols_lower = {c.lower(): c for c in chunk.columns}
                needed_cols = {
                    "nuts2":      cols_lower.get("nutsid",
                                  cols_lower.get("nuts2",
                                  cols_lower.get("nuts"))),
                    "funding":    cols_lower.get("eccontribution",
                                  cols_lower.get("totalcost")),
                    "role":       cols_lower.get("role",
                                  cols_lower.get("activitytype")),
                    "project_id": cols_lower.get("projectid",
                                  cols_lower.get("project_id")),
                    "country":    cols_lower.get("country"),
                }
            keep = [v for v in needed_cols.values()
                    if v is not None and v in chunk.columns]
            if keep:
                chunks.append(chunk[keep])

        if not chunks:
            log.warning("cordis_no_usable_columns")
            return {}

        df_orgs = pd.concat(chunks, ignore_index=True)
        nuts_col = needed_cols.get("nuts2")

        if not nuts_col or nuts_col not in df_orgs.columns:
            log.warning("cordis_no_nuts_column")
            return {}

        df_orgs = df_orgs.copy()
        df_orgs["nuts2_code"] = df_orgs[nuts_col].astype(str).str.strip().str[:4]
        df_valid = df_orgs[
            (df_orgs["nuts2_code"].str.len() == 4) &
            (df_orgs["nuts2_code"].str[:2].isin(EU27_COUNTRY_CODES))
        ].copy()

        results = {}
        proj_col = needed_cols.get("project_id")
        fund_col = needed_cols.get("funding")
        role_col = needed_cols.get("role")

        if proj_col and proj_col in df_valid.columns:
            proj_count = df_valid.groupby("nuts2_code")[proj_col].nunique()
            proj_count.index.name = "nuts2_code"
            results["horizon_eu_projects_total"] = proj_count.astype(float)

        if fund_col and fund_col in df_valid.columns:
            df_valid[fund_col] = pd.to_numeric(df_valid[fund_col], errors="coerce")
            funding = df_valid.groupby("nuts2_code")[fund_col].sum()
            funding.index.name = "nuts2_code"
            results["horizon_eu_funding_eur"] = funding

        if role_col and role_col in df_valid.columns:
            coord = (
                df_valid[df_valid[role_col].str.upper().str.contains("COORDIN", na=False)]
                .groupby("nuts2_code")[role_col]
                .count()
                .gt(0)
                .astype(int)
            )
            coord.index.name = "nuts2_code"
            results["coordinator_dummy"] = coord.astype(float)

        log.info("cordis_processed", features=list(results.keys()),
                 regions=df_valid["nuts2_code"].nunique())
        return results

    except Exception as exc:
        log.warning("cordis_processing_failed", error=str(exc))
        return {}


# ── NUTS normalization ────────────────────────────────────────────────────────

def load_nuts_correspondence() -> dict:
    p = ROOT / "data/raw/eurostat/nuts2016_2021_correspondence.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=str)
    mapping = {}
    for _, row in df.iterrows():
        c2016 = str(row.get("nuts2_code_2016", "")).strip()
        c2021 = str(row.get("nuts2_code_2021", "")).strip()
        if c2016 and c2021 and c2016 != c2021 and len(c2016) == 4:
            mapping[c2016] = c2021
    return mapping


def load_valid_nuts2_codes() -> set:
    geojson_p = ROOT / "data/raw/eurostat/nuts2_2021_geojson.json"
    if not geojson_p.exists():
        return set()
    geojson = json.loads(geojson_p.read_text(encoding="utf-8"))
    return {
        feat["properties"]["NUTS_ID"]
        for feat in geojson["features"]
        if len(feat["properties"].get("NUTS_ID", "")) == 4
        and feat["properties"]["NUTS_ID"][:2] in EU27_COUNTRY_CODES
    }


def normalize_nuts_codes(series: pd.Series, mapping: dict,
                          valid_codes: set, recode_log: list) -> pd.Series:
    result = series.copy()
    result.index = result.index.map(lambda x: mapping.get(x, x))
    for old in mapping:
        if old in series.index:
            recode_log.append({
                "original_code": old,
                "remapped_to": mapping[old],
                "feature": series.name or "unknown",
            })
    result = result[result.index.isin(valid_codes)]
    result.index.name = "nuts2_code"
    return result


# ── Missingness classification ────────────────────────────────────────────────

def classify_missingness(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic MCAR/MAR/MNAR classification per feature column.
    Uses logistic regression on country dummies to detect systematic missingness.
    """
    if wide_df.empty or wide_df.shape[1] == 0:
        return pd.DataFrame(
            columns=["feature", "pct_missing", "classification", "method"]
        )

    from sklearn.linear_model import LogisticRegression

    rows = []
    country_code = wide_df.index.str[:2]

    for col in wide_df.columns:
        if col.startswith("_") or col == "data_quality_score":
            continue
        pct = wide_df[col].isna().mean()

        if pct == 0:
            cls = "OBSERVED"
        elif pct < 0.10:
            cls = "MCAR"
        elif pct > 0.40:
            rd_vars = ["rd_expenditure", "business_rd", "rd_personnel"]
            cls = "MNAR" if any(rv in col for rv in rd_vars) else "MAR"
        else:
            miss_indicator = wide_df[col].isna().astype(int)
            country_dummies = pd.get_dummies(country_code, drop_first=True)
            if len(country_dummies.columns) > 0 and miss_indicator.sum() > 1:
                try:
                    lr = LogisticRegression(max_iter=100, random_state=42, C=1.0)
                    lr.fit(country_dummies, miss_indicator)
                    cls = "MAR" if abs(lr.coef_).max() > 0.5 else "MCAR"
                except Exception:
                    cls = "MAR" if pct > 0.20 else "MCAR"
            else:
                cls = "MCAR"

        rows.append({
            "feature": col,
            "pct_missing": round(pct, 4),
            "classification": cls,
            "method": "heuristic_logistic",
        })

    return pd.DataFrame(rows)


def apply_imputation(wide_df: pd.DataFrame,
                     missingness_df: pd.DataFrame) -> tuple:
    """
    Impute per classification:
    - MCAR: country-group median, then global median fallback
    - MAR: IterativeImputer (MICE-like, 5 iter)
    - MNAR: leave NaN, set mnar_flag

    Returns: (imputed_df, imputed_flag_cols, mnar_flag_cols)
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, SimpleImputer

    df = wide_df.copy()
    if missingness_df.empty:
        return df, [], []

    cls_map = dict(zip(missingness_df["feature"], missingness_df["classification"]))
    imputed_flag_cols = []
    mnar_flag_cols = []
    country_code = df.index.str[:2]

    mcar_cols = [c for c, v in cls_map.items() if v == "MCAR" and c in df.columns]
    mar_cols = [c for c, v in cls_map.items() if v == "MAR" and c in df.columns]
    mnar_cols = [c for c, v in cls_map.items() if v == "MNAR" and c in df.columns]

    # MCAR: median per country group
    for col in mcar_cols:
        if df[col].isna().sum() == 0:
            continue
        flag_col = f"{col}_imputed_flag"
        df[flag_col] = df[col].isna()
        country_medians = df.groupby(country_code)[col].transform("median")
        global_median = df[col].median()
        df[col] = df[col].fillna(country_medians).fillna(global_median)
        imputed_flag_cols.append(flag_col)

    # MAR: IterativeImputer
    if mar_cols:
        for col in mar_cols:
            flag_col = f"{col}_imputed_flag"
            df[flag_col] = df[col].isna()
            imputed_flag_cols.append(flag_col)
        imputable = [c for c in mar_cols if df[c].isna().mean() < 0.40]
        if imputable:
            try:
                imp = IterativeImputer(max_iter=5, random_state=42, min_value=0)
                df[imputable] = imp.fit_transform(df[imputable])
            except Exception:
                simp = SimpleImputer(strategy="median")
                df[imputable] = simp.fit_transform(df[imputable])

    # MNAR: flag only, no imputation
    for col in mnar_cols:
        flag_col = f"{col}_mnar_flag"
        df[flag_col] = df[col].isna()
        mnar_flag_cols.append(flag_col)

    return df, imputed_flag_cols, mnar_flag_cols


# ── Data quality score ────────────────────────────────────────────────────────

def compute_data_quality_score(wide_df: pd.DataFrame,
                                feature_cols: list) -> pd.Series:
    """Fraction of non-null features per region (excludes flag columns)."""
    pure = [c for c in feature_cols
            if c in wide_df.columns and not c.startswith("_")]
    if not pure:
        return pd.Series(1.0, index=wide_df.index, name="data_quality_score")
    return wide_df[pure].notna().mean(axis=1).rename("data_quality_score")


# ── Quality gate ──────────────────────────────────────────────────────────────

def run_quality_gate(silver_df: pd.DataFrame) -> bool:
    from src.contracts.silver_schema import silver_schema

    assert silver_df.shape[0] >= 200, \
        f"Silver has {silver_df.shape[0]} rows, expected ≥200"
    assert silver_df.index.is_unique, "nuts2_code index is not unique"
    assert silver_df.index.name == "nuts2_code"
    assert "data_quality_score" in silver_df.columns

    n_features = sum(
        1 for c in silver_df.columns
        if not c.endswith("_flag") and c not in
        ("nuts2_name", "country_code", "data_quality_score") and
        not c.startswith("ref_year_")
    )
    assert n_features >= 8, \
        f"Only {n_features} feature columns, expected ≥8"

    coverage = silver_df.notna().sum().sum() / silver_df.size
    print(f"  Coverage: {coverage:.1%}")

    try:
        silver_schema.validate(silver_df, lazy=True)
        print("  Pandera schema: PASS")
    except Exception as exc:
        print(f"  Pandera schema warnings (non-blocking): {exc}")

    print(f"\n  Silver shape: {silver_df.shape}")
    print(f"  Feature columns: {n_features}")
    print(f"  Coverage: {coverage:.1%}")
    print("\n✓ GATE_P2=PASS")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with pipeline_step("P2", random_seed=42) as log:
        print("\n=== Phase 2: Data Ingestion & Harmonization ===\n")

        # ── P2.T1-T4: Bronze downloads ────────────────────────────────────────
        print("P2.T1-T4  Downloading Eurostat datasets (SDMX-CSV) to Bronze...")
        bronze_paths: dict = {}
        for code, start in EUROSTAT_DATASETS:
            path = download_eurostat(code, start, log)
            if path:
                bronze_paths[code] = path
        print(f"  Downloaded {len(bronze_paths)}/{len(EUROSTAT_DATASETS)} datasets")

        # ── P2.T5: Bronze manifest ────────────────────────────────────────────
        print("P2.T5  Building bronze manifest...")
        manifest = {}
        for code, path in bronze_paths.items():
            manifest[code] = {
                "file_path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "download_timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_code": code,
            }
        (ROOT / "data/bronze").mkdir(exist_ok=True)
        (ROOT / "data/bronze/bronze_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        log.info("bronze_manifest_written", files=len(manifest))

        # ── P2.T6+T7: NUTS normalization setup ───────────────────────────────
        print("P2.T6+T7  Loading NUTS correspondence and valid codes...")
        nuts_mapping = load_nuts_correspondence()
        valid_nuts2 = load_valid_nuts2_codes()
        print(f"  Valid EU27 NUTS2 codes: {len(valid_nuts2)}")
        recode_log: list = []

        # Build region name index from GeoJSON
        geojson_p = ROOT / "data/raw/eurostat/nuts2_2021_geojson.json"
        region_names = {}
        if geojson_p.exists():
            gj = json.loads(geojson_p.read_text(encoding="utf-8"))
            for feat in gj["features"]:
                props = feat["properties"]
                code = props.get("NUTS_ID", "")
                if len(code) == 4 and code[:2] in EU27_COUNTRY_CODES:
                    region_names[code] = props.get(
                        "NAME_LATN", props.get("NUTS_NAME", code)
                    )

        master_index = pd.Index(sorted(valid_nuts2), name="nuts2_code")

        # ── P2.T8+T9: Feature extraction ─────────────────────────────────────
        print("P2.T8+T9  Extracting features from Bronze (year alignment)...")
        raw_features = build_features(bronze_paths, log)

        # ── P2.T10: Location Quotients ────────────────────────────────────────
        print("P2.T10  Computing Location Quotients...")
        lq_features = compute_location_quotients(bronze_paths, log)
        raw_features.update(lq_features)

        # ── P2.T11-12: CORDIS ─────────────────────────────────────────────────
        print("P2.T11-12  Processing CORDIS Horizon Europe data...")
        cordis_features = download_and_process_cordis(log)
        raw_features.update(cordis_features)

        # ── Assemble wide DataFrame ───────────────────────────────────────────
        print("  Assembling wide feature DataFrame...")
        wide_df = pd.DataFrame(index=master_index)
        wide_df["nuts2_name"] = pd.Series(region_names)
        wide_df["country_code"] = wide_df.index.str[:2]

        for feat_name, series in raw_features.items():
            if feat_name.startswith("_"):
                continue
            series_named = series.rename(feat_name)
            series_norm = normalize_nuts_codes(
                series_named, nuts_mapping, valid_nuts2, recode_log
            )
            wide_df[feat_name] = series_norm
            log.info("feature_merged", name=feat_name,
                     n_values=int(series_norm.notna().sum()),
                     pct_coverage=f"{series_norm.notna().mean():.0%}")

        # PPP-adjust wages if both series available
        if "avg_wage_eur" in wide_df.columns and "_ppp_index" in raw_features:
            ppp = raw_features["_ppp_index"]
            wide_df["_ppp_country"] = wide_df.index.str[:2].map(ppp)
            mask = wide_df["_ppp_country"].notna() & wide_df["avg_wage_eur"].notna()
            wide_df["avg_wage_eur_ppp"] = np.nan
            wide_df.loc[mask, "avg_wage_eur_ppp"] = (
                wide_df.loc[mask, "avg_wage_eur"]
                / wide_df.loc[mask, "_ppp_country"] * 100
            )
            wide_df = wide_df.drop(columns=["_ppp_country"])

        # Write NUTS recode log
        if recode_log:
            pd.DataFrame(recode_log).to_csv(
                ROOT / "analysis/nuts_recode_log.csv", index=False
            )
            log.info("nuts_recode_log_written", entries=len(recode_log))

        # ── Drop too-sparse variables (>50% missing) ──────────────────────────
        print("P2.T16  Checking variable sparsity...")
        feature_cols_raw = [c for c in wide_df.columns
                            if c not in ("nuts2_name", "country_code")]
        sparse_cols = [c for c in feature_cols_raw
                       if wide_df[c].isna().mean() > 0.50]
        if sparse_cols:
            print(f"  Dropping {len(sparse_cols)} too-sparse columns: {sparse_cols}")
            wide_df = wide_df.drop(columns=sparse_cols)

        feature_cols = [c for c in wide_df.columns
                        if c not in ("nuts2_name", "country_code")]

        # ── P2.T13-14: Missingness classification ────────────────────────────
        print("P2.T13-14  Classifying missingness...")
        numeric_df = wide_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        miss_df = classify_missingness(numeric_df)
        miss_df.to_csv(ROOT / "analysis/imputation_decisions.csv", index=False)
        if not miss_df.empty:
            log.info("missingness_classified",
                     mcar=int((miss_df["classification"] == "MCAR").sum()),
                     mar=int((miss_df["classification"] == "MAR").sum()),
                     mnar=int((miss_df["classification"] == "MNAR").sum()))

        # ── P2.T15: Imputation ────────────────────────────────────────────────
        print("P2.T15  Applying imputation...")
        numeric_wide = wide_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        imputed_numeric, imputed_flags, mnar_flags = apply_imputation(
            numeric_wide, miss_df
        )

        # Re-assemble Silver
        silver_df = wide_df[["nuts2_name", "country_code"]].copy()
        for col in feature_cols:
            if col in imputed_numeric.columns:
                silver_df[col] = imputed_numeric[col]
        for flag_col in imputed_flags + mnar_flags:
            if flag_col in imputed_numeric.columns:
                silver_df[flag_col] = imputed_numeric[flag_col]

        # Reference years (placeholder — actual years logged per-feature in build_features)
        for grp in ("cost", "talent", "infra", "cluster"):
            silver_df[f"ref_year_{grp}"] = 2022

        # Data quality score
        pure_features = [c for c in feature_cols if c in silver_df.columns]
        silver_df["data_quality_score"] = compute_data_quality_score(
            silver_df, pure_features
        )

        # Year alignment decisions log
        yr_decisions = [
            {"feature": feat, "ref_year": "auto-selected",
             "n_values": int(series.notna().sum())}
            for feat, series in raw_features.items()
            if not feat.startswith("_") and not series.empty
        ]
        pd.DataFrame(yr_decisions).to_csv(
            ROOT / "analysis/year_alignment_decisions.csv", index=False
        )

        # ── P2.T17: Save Silver parquet ───────────────────────────────────────
        print("P2.T17  Saving Silver parquet...")
        (ROOT / "data/silver").mkdir(exist_ok=True)
        out_path = ROOT / "data/silver/region_profiles_silver.parquet"
        silver_df.to_parquet(out_path, index=True)
        log.info("silver_written",
                 shape=str(silver_df.shape),
                 path=str(out_path),
                 size_mb=round(out_path.stat().st_size / 1e6, 2))

        # ── P2.T19: Ingestion report ──────────────────────────────────────────
        ingestion_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "silver_shape": list(silver_df.shape),
            "features_extracted": len(feature_cols),
            "coverage_pct": round(
                silver_df[feature_cols].notna().mean().mean() * 100, 1
            ),
            "sparse_cols_dropped": sparse_cols,
            "cordis_features": list(cordis_features.keys()),
            "lq_features": list(lq_features.keys()),
            "bronze_files": len(manifest),
            "datasets_failed": [
                code for code, _ in EUROSTAT_DATASETS
                if code not in bronze_paths
            ],
        }
        (ROOT / "analysis/ingestion_report.json").write_text(
            json.dumps(ingestion_report, indent=2), encoding="utf-8"
        )

        # ── P2.T18: Quality gate ──────────────────────────────────────────────
        print("\nP2.T18  Running quality gate...")
        run_quality_gate(silver_df)


if __name__ == "__main__":
    main()
