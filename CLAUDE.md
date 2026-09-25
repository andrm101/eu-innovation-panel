# CLAUDE.md — EU Regional Innovation Panel

## Project Purpose
Exploratory, non-causal descriptive analysis of EU NUTS 2 regional innovation ecosystems.
This project characterizes regional variation and produces a descriptive archetype taxonomy.
It does NOT establish causal relationships, make predictions, or prescribe policy.

## Non-Causal Framing (STRICT)
- This is observational, cross-sectional, descriptive research
- All findings are described as "associated with", "correlates with", "characterizes"
- NEVER use: causes, effect of X on Y, driven by, leads to, predicts (unless explicitly hedged)
- Use hedged language: "appears to correlate", "is associated with", "patterns consistent with"

## Forbidden Tokens (auto-enforced by vocab_guard.py)
The following patterns are NEVER permitted in any artifact, code comment, figure caption,
dashboard label, profile document, or manuscript passage:

- "best region" / "optimal location" / "top region" / "ranked"
- "causes" / "effect of X on Y" / "driven by" / "leads to"
- "we recommend locating" / "ideal for" / "suitable for investment"
- "site selection score" / "composite ranking"

Run `python src/utils/vocab_guard.py <file>` to check any artifact before committing.

## Primary Key
`nuts2_code` is the immutable primary key for all datasets.
Format: `^[A-Z]{2}[A-Z0-9]{2}$` (e.g., DE21, FR10, PL63)
NUTS 2021 classification is the canonical vintage. All older vintages must be mapped
to 2021 codes using `data/raw/eurostat/nuts2016_2021_correspondence.csv`.

## Data Sources (Eurostat-first policy)
All data must come from freely accessible sources. Priority:
1. Eurostat bulk downloads (primary for all EU regional statistics)
2. CORDIS CSV bulk download (Horizon Europe project data)
3. OECD.Stat bulk export (supplementary)
4. No EPO PATSTAT registration — use Eurostat `pat_ep_rtot` as patent proxy

## Seed Policy
- All scripts must call `np.random.seed(42)` at module top level
- All sklearn estimators: `random_state=42` explicitly set
- All randomness logged via `src/utils/logging_config.py` `pipeline_step()` context manager
- Seed check: `python src/utils/seed_check.py scripts/` must exit 0

## Imputation Documentation Policy
Every imputed value must have a corresponding `{feature}_imputed_flag` boolean column.
Every MNAR-classified variable must have a `{feature}_mnar_flag` column and MUST NOT be imputed.
All imputation decisions recorded in `analysis/imputation_decisions.csv`.

## Medallion Architecture
- Bronze: raw files as received, never modified, in `data/bronze/`
- Silver: harmonized, imputed, NUTS-aligned panel in `data/silver/`
- Gold: Silver + PCA scores + archetype assignments in `data/gold/`

## Naming Conventions
- Scripts: `p{phase}_{short_description}.py` e.g., `p2_ingest_harmonize.py`
- Notebooks: `nb_{phase}_{description}.ipynb` (exploration only, not production)
- Figures: `{phase}_{description}.png` e.g., `p3_lisa_rd_expenditure.png`
- Analysis artifacts: descriptive names, lowercase, underscores

## Paths
All paths in scripts must be relative to project root (`EU-Innovation-Panel/`).
Use `pathlib.Path(__file__).parent.parent` to resolve root; never hardcode absolute paths.

## Phase Status
| Phase | Status |
|---|---|
| P0 — Scaffold | Scaffolded (this file) |
| P1 — Data Inventory | PASS (40-entry catalog, 242 NUTS2 codes) |
| P2 — Ingestion | PASS (Silver: 242 × 43, 18 features, 100% coverage) |
| P3 — EDA | PASS (9 figures, Moran's I significant for 17/18 features, 7 components for 80% variance) |
| P4 — PCA | PASS (4 scores, 242/242 coverage; talent PC1=74.4% after hi_tech exclusion; cluster PC1=52.5%; 86 transition regions; Gold: 242×66) |
| P5 — Clustering | PASS (k=2 silhouette-optimal=0.409; A1: Catching-up & peripheral 108 regions; A2: Advanced innovation 134 regions; Ward ARI=0.615; gold schema PASSED) |
| P6 — Barriers/Enablers | PASS (16/16 FDR-sig; 13 large-effect features; L2 logistic acc=99.2%; top enablers: GDP/capita d=2.32, HRST d=2.06, enterprise internet d=1.81; LQ energy/utilities negatively associated d=-1.09; 5 high within-archetype CV features) |
| P6b — Spatial Autocorrelation | PASS (Global Moran's I: archetype=0.515, prosperity=0.751, talent=0.496, infra=0.697, cluster=0.341; all p=0.001; LISA: 82 HH / 0 LL / 21 HL / 18 LH / 121 NS — A2 clusters spatially, A1 dispersed) |
| P6c — Country Fixed Effects | WARN (scores retain 35–45% of effect after country-demeaning; demeaned logistic acc=70.7%; 14/27 countries mixed; talent features most within-country: HRST 43%, tertiary 55%, LQ KIS 63%) |
| P6d — Co-Specialisation Typology | PASS (10-pattern taxonomy from 4 LQ cols; 78.5% of regions specialised in ≥1 sector; 39.3% co-specialised in ≥2; dominant pattern: Energy/utilities only n=78 74% A1; Diversified innovation n=33 91% A2; dashboard Tab 6 added) |
| P7 — Dashboard | PASS (Streamlit app, 5 tabs; NUTS2 GeoJSON choropleth integrated; country choropleth in expander; runs on localhost:8501) |
| P8 — Profiles | PASS (2 archetype profiles in MD+JSON; 242-region table with uncertain_assignment flag; NUTS2 codes fixed; A1: 108 regions/16 countries/82 transition; A2: 134 regions/25 countries/4 transition) |
| P9 — Manuscript | PASS (4,341 words, 588 lines; YAML+abstract, 7 sections, 2 appendices; all numbers data-driven; GATE_P9=PASS) |
| P10 — Audit | PASS (2026-09-25: seed audit, schema import check, vocab guard, content-hash verification all green; two real bugs fixed along the way — p4_pca_scores.py had a false-positive-triggering `PCA(random_state=random_state)` passthrough the seed auditor couldn't statically resolve, and p8_profiles.py's archetype-profile template used the forbidden word "ranked". expected_hashes.json now covers data/gold/region_profiles_gold.parquet; profile PDFs and manuscript PDF are not yet rendered by this pipeline, so out of hash-audit scope for now — see REPRODUCE.md) |

## Key Design Decisions (locked)
- Reference year: per-variable, latest year with ≥60% NUTS2 coverage (lowered from 80% due to structural sparsity)
- Archetype count k: data-driven via silhouette score (k ≤ 5); human review if silhouette < 0.35
- Journal target: deferred to Phase 9 journal fit analysis
- Patent data: Eurostat `pat_ep_rtot` only (no EPO PATSTAT registration)
- RIS: use 2023 edition as external validation benchmark if 2025 not available
- LQ source: `htec_emp_reg2` (section-level proxies); `lfst_r_lfe2en2` returns empty from SDMX API
- Eurostat API: format=SDMX-CSV (confirmed working); format=TSV/bulk deprecated
- Known data gaps (documented in analysis/ingestion_report.json):
  - avg_wage_eur_ppp: earn_ses_pub2s national-only
  - horizon_eu_*: CORDIS URL 404 (relocated)
  - startup_density_proxy: bd_9bd_sz_cl_r2 returns 413
  - renewable_energy_share: nrg_r_rgen returns 404
