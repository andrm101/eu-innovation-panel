# Reproducing the EU Regional Innovation Panel Analysis

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
| Cluster diagnostics | `analysis/p5b_cluster_summary.csv` | Silhouette, ARI |
| Regional profiles (data) | `profiles/p8_archetype_profiles.md`, `.json`, `p8_region_table.csv` | Per-archetype writeups and a 242-region table |
| Manuscript | `reports/p9_manuscript.md` | 4,341-word draft (Markdown; not yet rendered to PDF) |
| Dashboard | Launch with `make dashboard` | localhost:8050 |

**Not yet produced by this pipeline:** per-region `profiles/<NUTS2_code>.pdf`
files and a rendered `manuscript/main.pdf` — P8/P9 currently stop at
Markdown/CSV/JSON output; PDF rendering is a genuinely open gap, not part of
this reproducibility audit's scope.

SHA-256 checksums: see `expected_hashes.json`, generated via
`python scripts/pipeline_audit.py --mode=generate-hashes` once a pipeline
output is trusted as the reference (currently covers the Gold parquet only,
for the reason above). Parquet files are hashed on canonical data content, not
raw file bytes — pyarrow embeds non-data metadata (e.g. write timestamps)
that differs between two numerically-identical runs, which would otherwise
cause the hash check to falsely fail on a reproducible pipeline.

**Generate the reference hash in the same environment you'll verify against.**
Content-hashing uses `df.to_csv()`, which is stable across repeated runs
*within* one pandas/pyarrow version but not guaranteed byte-identical *across*
different versions (e.g. host Python vs. the Docker image, or two Docker image
versions) — float formatting has changed between pandas releases before. This
was confirmed directly: hashes generated on the host and verified inside the
container (numerically-equal data, different pandas version) mismatched, while
hashes generated and verified within the same container image matched. Always
regenerate `expected_hashes.json` from inside `eu-innov:latest` if you intend
to verify future container runs, and separately from the host if you intend to
verify host-run (`make pipeline` without Docker) reproductions.

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
