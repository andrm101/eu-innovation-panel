# CORDIS Raw Data — Source Documentation

All files in this directory are immutable originals. Do not modify.

## Source
European Commission CORDIS (Community Research and Development Information Service)

## Dataset
**Horizon Europe Projects** — all funded projects 2021–2027

- Bulk CSV download: https://data.europa.eu/data/datasets/cordisprojects
- Direct CSV (projects): `https://cordis.europa.eu/data/cordis-h2020projects.csv` (H2020)
- Direct CSV (HE): `https://cordis.europa.eu/data/cordis-HEprojects.csv` (Horizon Europe)
- License: CC-BY-4.0

## Variables Derived in Phase 2
- `horizon_eu_projects_total`: count of projects per NUTS 2 region (beneficiary location)
- `horizon_eu_funding_eur`: total EU contribution per NUTS 2 region
- `coordinator_dummy`: 1 if region hosts ≥1 project coordinator

## Geocoding Method
Beneficiary postcodes/cities mapped to NUTS 2 via GISCO geocoder REST API:
`https://gisco-services.ec.europa.eu/tools/geofusion/`
Fallback: NUTS_ID field where directly available in CORDIS export.

## Known Limitations
- Geocoding accuracy ~85-90%; remaining ~10-15% assigned to country-level centroid NUTS 2
- Multi-country projects counted in each participating region's totals
- Coverage: Horizon Europe 2021-2027 only (not H2020 for consistency)
