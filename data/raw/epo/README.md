# Patent Data — Source Documentation

## Decision
EPO PATSTAT BDDS registration was evaluated and **not pursued**.
**Chosen substitute**: Eurostat dataset `pat_ep_rtot` (EPO patent applications by NUTS 2 region).

**Rationale**: Eurostat `pat_ep_rtot` provides EPO patent applications per million population
aggregated to NUTS 2, freely accessible without registration, with complete EU27 coverage.
PATSTAT would add applicant-level geocoding precision but is unnecessary for regional-level
descriptive analysis.

## Chosen Source
- **Dataset code**: `pat_ep_rtot`
- **URL**: https://ec.europa.eu/eurostat/databrowser/view/pat_ep_rtot/
- **License**: CC-BY-4.0
- **Variable**: EPO patent applications per million population, by inventor NUTS 2 region
- **Reference year**: Latest available (2020 as of 2024)
- **Known limitation**: Uses inventor address (not applicant); ~3-4 year filing lag

## PATSTAT Reference (for future work)
EPO PATSTAT bulk data (BDDS) has been freely available since January 2025:
- Registration: https://www.epo.org/en/searching-for-patents/data/bulk-data-sets/patstat
- Would enable: applicant-to-NUTS-2 geocoding, citation analysis, technology class breakdown
- Recommended for future causal panel data extension of this study
