# OECD Regional Database — Source Documentation

All files in this directory are immutable originals. Do not modify.

## Source
OECD Regional Statistics — REGION_INNOVATION dataset

- Access via OECD.Stat bulk export: https://stats.oecd.org/Index.aspx?DataSetCode=REGION_INNOVATION
- Manual download required: select TL2 (equivalent to NUTS 2) geographic level
- License: CC-BY-4.0 (OECD open data)

## Role in This Project
SUPPLEMENTARY only — used as cross-check against Eurostat figures.
OECD TL2 regions ≈ NUTS 2 but correspondence is not always 1:1.
The TL2→NUTS2 mapping is documented in `analysis/oecd_tl2_nuts2_mapping.csv` (Phase 2).

## Variables of Interest
- Innovation inputs (R&D expenditure, personnel)
- Innovation outputs (patents, trademarks)
- Regional economic indicators

## Download Instructions
1. Go to https://stats.oecd.org/Index.aspx?DataSetCode=REGION_INNOVATION
2. Select: Geographic level = TL2, Country = EU27 members
3. Export as CSV
4. Save to: `data/raw/oecd/REGION_INNOVATION_TL2_EU27.csv`
