# Eurostat Raw Data — Source Documentation

All files in this directory are immutable originals. Do not modify.

**Downloaded**: will be filled per file in `bronze_manifest.json`
**NUTS vintage**: 2021 (NUTS2016 codes recoded via correspondence table)

## Dataset Codes Used

| Dataset Code | Feature | Description |
|---|---|---|
| [tgs00010](https://ec.europa.eu/eurostat/databrowser/view/tgs00010/) | `gdp_per_capita_pps` | GDP per capita in Purchasing Power Standards (PPS), NUTS 2... |
| [earn_ses_pub2s](https://ec.europa.eu/eurostat/databrowser/view/earn_ses_pub2s/) | `avg_wage_eur_ppp` | Mean monthly earnings by NUTS 2 region, adjusted to EUR PPP via prc_ppp_ind... |
| [prc_ppp_ind](https://ec.europa.eu/eurostat/databrowser/view/prc_ppp_ind/) | `price_level_index` | Regional price level indices for GDP deflation to PPP; used as denominator for m... |
| [hrst_st_rcat](https://ec.europa.eu/eurostat/databrowser/view/hrst_st_rcat/) | `hrst_per_1000` | Human Resources in Science and Technology (HRST) as share of active population p... |
| [educ_uoe_enrt07](https://ec.europa.eu/eurostat/databrowser/view/educ_uoe_enrt07/) | `tertiary_enrolment_rate` | Students enrolled in tertiary education (ISCED 5-8) as % of 20-24 age group, NUT... |
| [tgs00007](https://ec.europa.eu/eurostat/databrowser/view/tgs00007/) | `employment_rate` | Employment rate (20–64 years) by NUTS 2 region, %... |
| [rd_p_persreg](https://ec.europa.eu/eurostat/databrowser/view/rd_p_persreg/) | `rd_personnel_pct_employment` | R&D personnel and researchers as % of total employment, NUTS 2... |
| [educ_uoe_grad02](https://ec.europa.eu/eurostat/databrowser/view/educ_uoe_grad02/) | `tertiary_graduates_stem_pct` | Graduates in STEM fields (ISCED-F 05-08) as % of total tertiary graduates... |
| [demo_r_gind3](https://ec.europa.eu/eurostat/databrowser/view/demo_r_gind3/) | `net_migration_rate` | Net migration rate per 1000 inhabitants, NUTS 2 — proxy for talent attractivenes... |
| [isoc_r_broad_h](https://ec.europa.eu/eurostat/databrowser/view/isoc_r_broad_h/) | `broadband_penetration_pct` | Households with broadband internet access, % of all households, NUTS 2... |
| [urb_ctrans](https://ec.europa.eu/eurostat/databrowser/view/urb_ctrans/) | `urban_transport_index` | Urban transport infrastructure indicators (Urban Audit) — public transport stops... |
| [tran_r_viaflcu](https://ec.europa.eu/eurostat/databrowser/view/tran_r_viaflcu/) | `transport_infrastructure_density` | Length of motorways and railways per 1000 km² area, NUTS 2... |
| [isoc_r_iuse_i](https://ec.europa.eu/eurostat/databrowser/view/isoc_r_iuse_i/) | `internet_use_enterprises_pct` | Enterprises using the internet, % of enterprises with ≥10 employees, NUTS 2... |
| [rd_e_gerdreg](https://ec.europa.eu/eurostat/databrowser/view/rd_e_gerdreg/) | `rd_expenditure_pct_gdp` | Total intramural R&D expenditure (GERD) as % of GDP, NUTS 2... |
| [tgs00063](https://ec.europa.eu/eurostat/databrowser/view/tgs00063/) | `business_rd_pct_gdp` | Business enterprise R&D expenditure (BERD) as % of GDP, NUTS 2... |
| [pat_ep_rtot](https://ec.europa.eu/eurostat/databrowser/view/pat_ep_rtot/) | `epo_patents_per_mio_pop` | EPO patent applications per million population by inventor NUTS 2 region... |
| [yth_empl_040](https://ec.europa.eu/eurostat/databrowser/view/yth_empl_040/) | `youth_employment_rate` | Employment rate of young people aged 15-29, NUTS 2, % — proxy for workforce pipe... |
| [bd_9bd_sz_cl_r2](https://ec.europa.eu/eurostat/databrowser/view/bd_9bd_sz_cl_r2/) | `startup_density_proxy` | Business births (newly born enterprises) per 10,000 persons employed, NUTS 2 — p... |
| [nrg_r_rgen](https://ec.europa.eu/eurostat/databrowser/view/nrg_r_rgen/) | `renewable_energy_share` | Share of renewable energy in gross final energy consumption, NUTS 2, %... |
| [lfst_r_lfe2en2](https://ec.europa.eu/eurostat/databrowser/view/lfst_r_lfe2en2/) | `employment_by_nace` | Employment by economic activity (NACE Rev.2) and sex, NUTS 2 — used to compute a... |
| [lfst_r_lfe2en2](https://ec.europa.eu/eurostat/databrowser/view/lfst_r_lfe2en2/) | `lq_nace_j62j63` | Location Quotient for NACE J62+J63 (IT services + data processing) — AI/ML secto... |
| [lfst_r_lfe2en2](https://ec.europa.eu/eurostat/databrowser/view/lfst_r_lfe2en2/) | `lq_nace_c21_m72` | Location Quotient for NACE C21 (pharmaceutical manufacturing) + M72 (R&D service... |
| [lfst_r_lfe2en2](https://ec.europa.eu/eurostat/databrowser/view/lfst_r_lfe2en2/) | `lq_nace_c26` | Location Quotient for NACE C26 (manufacture of computer, electronic and optical ... |
| [lfst_r_lfe2en2 + nrg_r_rgen](https://ec.europa.eu/eurostat/databrowser/view/lfst_r_lfe2en2 + nrg_r_rgen/) | `lq_nace_d35_clean` | Composite Location Quotient for NACE D35 (electricity, gas supply) weighted by r... |
| [nama_10r_3gva](https://ec.europa.eu/eurostat/databrowser/view/nama_10r_3gva/) | `gva_hi_tech_services_share` | GVA share of high-tech knowledge-intensive services (NACE J+M+N subset), NUTS 2 ... |
| [htec_emp_reg2](https://ec.europa.eu/eurostat/databrowser/view/htec_emp_reg2/) | `hi_tech_employment_pct` | Employment in high-technology sectors as % of total employment, NUTS 2... |
| [demo_r_pjanaggr3](https://ec.europa.eu/eurostat/databrowser/view/demo_r_pjanaggr3/) | `population` | Total population by NUTS 2 region (January 1 reference date) — used as denominat... |
| [demo_r_d3dens](https://ec.europa.eu/eurostat/databrowser/view/demo_r_d3dens/) | `population_density` | Population density (inhabitants per km²), NUTS 2 — agglomeration proxy... |

## Access Method
Eurostat SDMX REST API:
```
https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{CODE}
    ?format=TSV&compressed=false&lang=EN
```

## License
All Eurostat data: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
Citation: Eurostat (<year>), [Dataset code], accessed <date>