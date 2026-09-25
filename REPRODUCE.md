# Reproducing the EU Regional Innovation Panel Analysis

> This file will be completed in Phase 10. Stubs are provided now so the
> structure is committed from the start.

## 1. Prerequisites

- Docker Desktop ≥ 4.28 (free for personal/academic use)
- 16 GB RAM minimum (32 GB recommended for Phase 3 spatial operations)
- 20 GB free disk space
- Git

No API keys or paid accounts are required. All data sources are freely accessible.

## 2. Full Reproduction (Docker — recommended)

```bash
git clone <repo_url> EU-Innovation-Panel
cd EU-Innovation-Panel

# Build image (~15 min on first run; layers cached on rebuild)
docker build -t eu-innov:latest .

# Place raw data in data/raw/ (see Phase 1 instructions for download steps)
# Then run the full pipeline:
make reproduce
```

Expected output: all artifacts in `data/gold/`, `profiles/`, `analysis/`, `manuscript/`.

## 3. Expected Outputs

| Artifact | Path | Notes |
|---|---|---|
| Silver panel | `data/silver/region_profiles_silver.parquet` | 230-240 rows × 40+ cols |
| Gold panel | `data/gold/region_profiles_gold.parquet` | + PCA scores + archetypes |
| Cluster diagnostics | `analysis/clustering_metrics.json` | Silhouette, ARI, CH, DB |
| Regional profiles | `profiles/<NUTS2_code>.pdf` | 12-15 files |
| Manuscript | `manuscript/main.pdf` | Draft |
| Dashboard | Launch with `make dashboard` | localhost:8050 |

SHA-256 checksums: see `expected_hashes.json` (populated in Phase 10).

## 4. Estimated Runtime per Phase

| Phase | Estimated time |
|---|---|
| P1 Data Inventory | ~10 min |
| P2 Ingestion | ~30-45 min |
| P3 EDA | ~20 min |
| P4 PCA | ~5 min |
| P5 Clustering | ~10 min |
| P6 Barriers/Enablers | ~5 min |
| P8 Profiles | ~10 min (parallel) |
| **Total** | **~90-120 min** |

## 5. Partial Reproduction (non-Docker)

Requires Ubuntu 22.04+ or WSL2 on Windows with:
- Python 3.11 (`python3.11`)
- GDAL 3.8+ (`gdal-bin libgdal-dev`)
- PROJ 9.3+ (installed with GDAL)
- pandoc 3.1+ (download from github.com/jgm/pandoc/releases)
- TeX Live: `texlive-xetex texlive-fonts-recommended`

```bash
pip install uv
uv sync
make pipeline
```

## 6. Verifying Success

```bash
make audit
```

A passing audit prints `Audit complete.` with no errors.
The file `pipeline_audit.json` should contain `"overall_status": "PASS"`.
