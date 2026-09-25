import numpy as np
import pandas as pd
from analysis.p11_archetype_growth_premium import fit_growth_premium


def _synthetic_panel(n_per_archetype=30, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    # A2 regions grow systematically faster than A1 by a known true premium
    true_premium_pp = 1.2  # percentage points/year
    for archetype, base_growth in [("A1", 1.0), ("A2", 1.0 + true_premium_pp)]:
        for i in range(n_per_archetype):
            gdp_start = rng.uniform(15_000, 40_000)
            noise = rng.normal(0, 0.3)
            annual_growth_pct = base_growth + noise
            n_years = 11
            gdp_end = gdp_start * (1 + annual_growth_pct / 100) ** n_years
            rows.append({
                "nuts2_code": f"{archetype}_{i:03d}",
                "archetype_id": archetype,
                "gdp_pc_start": gdp_start,
                "gdp_pc_end": gdp_end,
                "n_years": n_years,
                "population": rng.uniform(200_000, 2_000_000),
            })
    return pd.DataFrame(rows)


def test_fit_growth_premium_recovers_known_effect():
    df = _synthetic_panel()
    out = fit_growth_premium(df)
    a2_row = out[out["archetype_id"] == "A2"].iloc[0]
    # True premium is 1.2pp/yr; regression should recover it within noise tolerance
    assert 0.8 < a2_row["growth_premium_pp"] < 1.6
    assert a2_row["ci_lo"] < a2_row["growth_premium_pp"] < a2_row["ci_hi"]


def test_fit_growth_premium_reports_n_regions():
    df = _synthetic_panel(n_per_archetype=25)
    out = fit_growth_premium(df)
    assert out["n_regions"].sum() == 50
