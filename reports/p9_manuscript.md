---
title: >
  Characterising EU NUTS2 Regional Innovation Ecosystems:
  A Descriptive Archetype Taxonomy and Spatial Analysis
author: Andrei Manoloiu
affiliation: MSc Data Science, University of Southern Denmark
date: May 2026
abstract: |
  Regional innovation capacity in the European Union exhibits persistent
  structural heterogeneity across NUTS2 regions. This paper characterises
  that heterogeneity using a data-driven archetype taxonomy derived from
  242 EU NUTS2 regions across all 27 member states. Composite
  dimension scores for prosperity, talent, digital infrastructure, and
  innovation cluster intensity are constructed via principal component
  analysis (PCA) and combined through k-means clustering (k=2,
  silhouette=0.409). Two structurally distinct archetypes emerge:
  *Catching-up & peripheral regions* (n=108, 45%
  of the sample, 82 transition regions) and
  *Advanced innovation regions* (n=134, 55%,
  4 transition regions). All 16 features distinguish archetypes at
  FDR-corrected significance (Benjamini-Hochberg, α=0.05), with large effect
  sizes (Cohen's d ≥ 0.8) for 13 features. GDP per capita
  (d=+2.32), human capital
  (d=+2.06),
  and enterprise digital adoption
  (d=+1.81)
  show the strongest associations. Spatial analysis confirms strong
  autocorrelation in archetype membership (Global Moran's I = 0.515,
  p < 0.001), with 82 spatially coherent advanced-innovation clusters
  and no significant catching-up clusters. Country fixed-effects
  decomposition shows that archetype separation persists within countries
  (within-country d ≥ 0.67 for all four dimension scores), though
  approximately half the gross effect is attributable to between-country
  differences. A co-specialisation typology across four sector dimensions
  reveals that energy-utility over-representation without innovation
  co-specialisation is the dominant structural profile of catching-up
  regions (n=78, 74% A1). All findings are observational and descriptive;
  no causal claims are made.
keywords: regional innovation systems, NUTS2, cluster analysis, spatial autocorrelation, EU cohesion policy, location quotient
---

# 1. Introduction

Regional innovation capacity is not uniformly distributed across the European
Union. Decades of structural funds policy and smart specialisation strategies
have not eliminated the persistent gap between frontier innovation economies
concentrated in northern and western Europe and the catching-up periphery
spanning southern and eastern member states (Eurostat, 2023a).
Understanding the structural characteristics that co-vary with these differences
— without asserting causal mechanisms — is a precondition for evidence-based
regional policy design.

This paper makes three contributions. First, it constructs a validated
two-archetype taxonomy of EU NUTS2 regional innovation ecosystems from
a harmonised Eurostat panel covering 242 regions and 18 structural
features. Second, it quantifies the spatial structure of archetype membership,
testing whether advanced innovation clusters are geographically coherent
or dispersed. Third, it decomposes archetype separation into between-country
and within-country components, assessing the extent to which observed
differences reflect national-level rather than regional-level structural
variation. A supplementary co-specialisation analysis characterises regional
sectoral profiles across four innovation-relevant sector dimensions,
linking structural LQ patterns to archetype membership.

The analysis is explicitly non-causal and cross-sectional. Data reflect the
latest available reference year with NUTS2 coverage for each variable
(2019–2022 depending on dataset). All associations are described using
hedged language consistent with observational methodology.

# 2. Conceptual Background

## 2.1 Regional Innovation Systems

The regional innovation systems (RIS) framework, developed by Cooke (1992)
and extended by Asheim and Cooke (1999), posits that innovation is shaped
by the institutional, industrial, and knowledge infrastructure of the region
in which it occurs. Rather than treating innovation as a firm-level phenomenon,
the RIS approach emphasises systemic relationships between firms, universities,
public agencies, and financial institutions embedded in specific territorial
contexts. Regions differ not only in the quantity of innovative activity
but in the structural composition of their innovation systems — the types
of knowledge they produce, the industries they anchor, and the governance
arrangements they operate under.

A descriptive taxonomy of EU NUTS2 regions by structural innovation profile
complements this literature by providing a systematic, data-driven
characterisation of structural variation across the full EU27 population,
rather than relying on case studies or pre-defined typologies.

## 2.2 EU Regional Divergence and Cohesion Policy

EU Cohesion Policy distinguishes three categories of regions by GDP per
capita relative to the EU27 average: less developed (<75%), transition
(75–90%), and more developed (>90%). Of the 242 regions in this
analysis, 86 are classified as transition or less developed under
this criterion (GDP < 75% of sample mean). The concentration of these regions
in the A1 archetype (82 of 108, 75.9%)
confirms that structural innovation capacity co-varies closely with the
convergence criteria used to target EU Structural Funds.

## 2.3 Spatial Econometrics and Innovation Clustering

Spatial dependence in economic variables is the norm rather than the exception
at the regional level (Anselin, 1995). Marshall-Arrow-Romer externalities,
labour market pooling, and knowledge spillovers operate over limited distances,
generating spatial autocorrelation in innovation outcomes (Feldman, 1994;
Audretsch and Feldman, 1996). Whether archetype membership itself exhibits
spatial clustering — i.e., whether advanced innovation regions tend to
neighbour other advanced regions — is an empirical question addressed
in Section 4.4 of this paper.

# 3. Data

## 3.1 Geographic Unit and Vintage

The unit of analysis is the NUTS 2021 Level 2 region (NUTS2). All
27 EU member states are represented. 242 regions are included
after excluding NUTS2 codes present in the Eurostat classification
but absent from at least 60% of the feature datasets (the minimum
coverage threshold applied at ingestion). The NUTS 2021 correspondence
table from Eurostat was used to align data from older vintages.

## 3.2 Feature Set

Eighteen structural features are included in the Silver layer,
spanning four conceptual dimensions:

| Dimension | Features |
|-----------|----------|
| Prosperity | GDP per capita (PPS) |
| Talent | HRST per 1,000; tertiary attainment rate; employment rate |
| Digital infrastructure | Broadband penetration (%); enterprise internet use (%) |
| Innovation cluster | GERD (% GDP); BERD (% GDP); EPO patents per million population; GVA ICT share; LQ ICT services (J62/J63); LQ KIS hi-tech (C21/M72); LQ hi-tech manufacturing (C26); LQ energy/utilities (D35) |

All data are sourced from Eurostat bulk downloads (SDMX-CSV format, API
accessed 2024). The reference year for each variable is the latest
available with ≥60% NUTS2 coverage. For R&D expenditure data, this
threshold was lowered from the standard 80% due to structural sparsity
in national R&D reporting at the NUTS2 level, a known characteristic
of Eurostat's `rd_e_gerdreg` dataset (Eurostat, 2023b).

## 3.3 Known Data Gaps

Four planned features could not be sourced at NUTS2 resolution:
average wage in PPP terms (available nationally only in `earn_ses_pub2s`),
Horizon Europe participation data (CORDIS API URL relocated, returning 404),
startup density (dataset `bd_9bd_sz_cl_r2` exceeds size limits without
key filters), and renewable energy generation share (`nrg_r_rgen` returned 404
at time of ingestion). These gaps are documented in `analysis/ingestion_report.json`.
Net migration rate was excluded from scoring after confirming a spurious
near-perfect correlation (r ≈ 1.0) with population in the Silver layer,
indicating a data construction issue in the source dataset.

## 3.4 Imputation

All missing values were imputed using median imputation within
the full NUTS2 panel. Variables classified as Missing Not At Random (MNAR)
— primarily R&D expenditure in regions with no reported R&D activity —
were imputed with a structural-zero proxy (0.01% GDP) following the
convention used in the Regional Innovation Scoreboard (Hollanders et al.,
2023). Every imputed value has a corresponding boolean `_imputed_flag`
column in the Gold layer. `hi_tech_employment_pct` was excluded from
scoring at the PCA stage after 100% of its values were found to be
imputed to the median (≈14.5%), rendering it analytically uninformative.

# 4. Methodology

## 4.1 Dimension Score Construction

Four composite dimension scores are constructed from the Silver layer:

**Prosperity score** (`score_cost`): Z-score of log₁p-transformed GDP per
capita in PPS. The log₁p transformation reduces right-skew (skewness = 1.42
in the untransformed distribution).

**Talent score** (`score_talent`): First principal component of three talent
features (HRST per 1,000; tertiary attainment rate; employment rate),
explaining 74.4% of variance in the talent feature block.

**Digital infrastructure score** (`score_infra`): Mean z-score of broadband
penetration and enterprise internet use. A simple mean was preferred over PCA
given the two-feature block — PCA PC1 would be equivalent to the mean under
equal covariance.

**Innovation cluster score** (`score_cluster`): First principal component of
eight innovation features (GERD, BERD, EPO patents, GVA ICT share,
four sector LQs), explaining 52.5% of variance in the innovation block.

Log₁p transformation was applied to nine features with skewness > 1.0
prior to PCA and score construction. Employment rate (skewness = −1.08)
was used untransformed. Regions with GDP per capita below 75% of the
EU27 sample mean are flagged as transition regions
(`is_transition_region`; 86 of 242 regions, 35.5%).

## 4.2 Clustering

Prior to clustering, all four dimension scores are re-standardised to unit
variance (mean 0, SD 1) to ensure equal weighting across dimensions that
differ in raw scale. K-means clustering (scikit-learn, `n_init=50`,
`max_iter=500`, `random_state=42`) is applied across k = 2 to 5. The
optimal k is selected by maximising the average silhouette coefficient
(Rousseeuw, 1987) subject to a minimum threshold of 0.35 and k ≤ 5.

Robustness is assessed using Ward's hierarchical agglomerative clustering
(scikit-learn `AgglomerativeClustering`, `linkage='ward'`). Agreement
between k-means and Ward solutions is quantified by the Adjusted Rand Index
(Hubert and Arabie, 1985). Calinski-Harabász (CH) and Davies-Bouldin (DB)
indices provide supplementary validity information.

| k | Silhouette | CH | DB | ARI (Ward) |
|---|-----------|----|----|------------|
| 2 | **0.409** | **258.1** | **0.896** | 0.615 |
| 3 | 0.338 | 231.0 | 0.997 | 0.815 |
| 4 | 0.241 | 185.0 | 1.237 | 0.454 |
| 5 | 0.229 | 160.6 | 1.376 | 0.507 |

k = 2 is selected as the optimal solution: it yields the highest silhouette
(0.409), CH index, and lowest DB index. k = 3 falls marginally below the
silhouette threshold (0.338 < 0.35). The Ward ARI of 0.615 at k = 2 indicates
moderate cross-method agreement; boundary disagreement at k = 2 is expected
given that the two solutions differ only in the treatment of the interface zone
between archetypes.

## 4.3 Feature Association Analysis

Feature associations with archetype membership are quantified using
Cohen's d (standardised mean difference, A2 − A1; Cohen, 1988) with
999-iteration percentile bootstrap 95% confidence intervals. The
Mann-Whitney U test (non-parametric, no normality assumption) provides
a complementary significance test; p-values are corrected for multiple
comparisons using the Benjamini-Hochberg false discovery rate procedure
(Benjamini and Hochberg, 1995; α = 0.05). Effect magnitudes follow
Cohen's (1988) conventions: |d| < 0.2 negligible, 0.2–0.5 small,
0.5–0.8 medium, > 0.8 large.

Discriminative validity is further assessed using L2-regularised logistic
regression (sklearn `LogisticRegression`, C = 0.5, solver = 'lbfgs';
Tibshirani, 1996). L2 regularisation is required due to near-complete
separation arising from the strong multivariate feature differences between
archetypes; unregularised maximum likelihood fails to converge.

## 4.4 Spatial Autocorrelation

Spatial dependence in archetype membership and dimension scores is assessed
using Moran's I (Moran, 1950) computed on Queen contiguity spatial weights
(libpysal 4.14, row-standardised). Queen contiguity is used in preference
to distance-based weights because NUTS2 regions vary substantially in area
(from Malta, 316 km², to Lappi, 98,984 km²), making distance-band weights
scale-sensitive. Statistical significance is assessed via 999-permutation
conditional randomisation (Anselin, 1995).

Local Indicators of Spatial Association (LISA; Anselin, 1995) identify
statistically significant local clusters (HH: high surrounded by high;
LL: low surrounded by low) and spatial outliers (HL, LH) at p ≤ 0.05.
Regions with no neighbours under the Queen criterion (n = 21, predominantly
island NUTS2 units) are excluded from spatial statistics and flagged.

## 4.5 Country Fixed-Effects Decomposition

To assess whether archetype separation reflects regional heterogeneity
within countries — as opposed to between-country differences — all dimension
scores and features are country-demeaned (each region's value minus its
country mean). Cohen's d is then recomputed on the demeaned values. The
percentage of the original effect retained after demeaning provides a
decomposition of effect size into within- and between-country components.
A demeaned logistic regression (features → archetype, no explicit country
dummies; country FE implicit in demeaning) assesses whether regional features
retain discriminative power after country variance removal.

## 4.6 Sectoral Co-Specialisation Typology

A rule-based typology classifies each region by which combination of four
sector location quotients exceeds LQ > 1.0 (over-represented relative to
the EU reference mean). The four dimensions are: ICT services (J62/J63),
knowledge-intensive services hi-tech (C21/M72), hi-tech manufacturing (C26),
and energy/utilities (D35). The hierarchy of 10 patterns is:
(1) Diversified innovation (J, K, C all > 1.0);
(2) Digital knowledge economy (J, K > 1.0);
(3) Digital-industrial (J, C > 1.0);
(4) Advanced manufacturing & R&D (K, C > 1.0);
(5) Energy-anchored mixed (D > 1.0 and any of J/K/C > 1.0);
(6–8) Single-sector specialisations (J only, K only, C only);
(9) Energy/utilities only (D > 1.0);
(10) Non-specialised (none > 1.0).

# 5. Results

## 5.1 Archetype Taxonomy

The k = 2 solution identifies two structurally distinct archetypes
(mean silhouette = 0.409), labelled based on their centroid profiles:

**Archetype 1 — Catching-up & peripheral regions** (n = 108,
45% of all NUTS2 regions).
82 regions (75.9%)
meet the transition region criterion. A1 spans 16 member states,
with the largest national representation from Italy (17),
Poland (15), and France (14).
The mean GDP per capita is €23,569 PPS,
representing 62.7% of the EU27
sample mean (€37,569 PPS). Mean GERD is
0.95% of GDP, against an EU27 mean
of 1.67%.

**Archetype 2 — Advanced innovation regions** (n = 134,
55% of all NUTS2 regions).
Only 4 regions (3.0%)
are transition regions. A2 spans 25 member states,
with the largest representation from Germany (36),
France (13), and Spain (12).
Mean GDP per capita is €48,853 PPS
(130.0% of EU27 mean).
Mean GERD is 2.24% of GDP.

**Dimension score profiles** (normalised [0,1]):

| Dimension | A1 mean | A2 mean | EU27 mean |
|-----------|---------|---------|-----------|
| Prosperity | 0.304 | 0.602 | 0.469 |
| Talent | 0.314 | 0.601 | 0.472 |
| Digital infra. | 0.413 | 0.663 | 0.551 |
| Innovation cluster | 0.325 | 0.631 | 0.494 |

The mean silhouette coefficient is 0.3993 for A1
and 0.4176 for A2. One region has a negative
silhouette coefficient (indicating possible misassignment); 37 regions
have silhouette < 0.20 and are flagged as `uncertain_assignment` in
the region table.

## 5.2 Feature Associations

All 16 features show statistically significant differences between archetypes
after FDR correction (BH α = 0.05). Thirteen features show large effects
(|d| ≥ 0.8). The five strongest associations with A2 membership are:

| Feature | Cohen's d | 95% CI | Interpretation |
|---------|-----------|--------|---------------|
| GDP per capita (PPS) | +2.320 | [2.017, 2.714] | Higher in A2 |
| HRST per 1,000 pop. | +2.063 | [1.825, 2.383] | Higher in A2 |
| Enterprise internet use (%) | +1.814 | [1.599, 2.075] | Higher in A2 |
| Tertiary attainment rate (%) | +1.617 | [1.388, 1.903] | Higher in A2 |
| EPO patents per million pop. | +1.490 | [1.187, 1.833] | Higher in A2 |

The only feature showing higher values in A1 with large effect size is the
location quotient for energy and utilities (D35): d = -1.086
[-1.376,
-0.845],
indicating that over-representation in energy production and distribution
co-varies with catching-up status. The L2-regularised logistic regression
achieves 99.2% accuracy on the training sample, confirming near-complete
multivariate separation.

Five features exhibit high within-archetype coefficients of variation
(CV > 0.50), most notably population density (CV = 3.37 in A1),
EPO patents per million (CV = 1.24 in A1), and BERD (CV = 0.82 in A1),
indicating that the two-archetype solution masks substantial internal
heterogeneity, particularly within A1.

## 5.3 Spatial Autocorrelation

Global Moran's I is statistically significant (p < 0.001, 999 permutations)
for all five tested variables. Spatial autocorrelation is strongest for
the prosperity score (I = 0.7508) and
digital infrastructure score (I = 0.6969),
indicating that contiguous NUTS2 regions tend to resemble each other more
than randomly permuted regions. Archetype membership shows substantial
spatial clustering (I = 0.5150), consistent with
the geographic concentration of advanced innovation regions in Germany,
Benelux, and northern France.

| Variable | Moran's I | p (permutation) |
|----------|-----------|-----------------|
| Archetype (0/1) | 0.5150 | 0.0010 |
| Prosperity score | 0.7508 | 0.0010 |
| Talent score | 0.4962 | 0.0010 |
| Digital infra. score | 0.6969 | 0.0010 |
| Innovation cluster score | 0.3411 | 0.0010 |

LISA analysis identifies 82 statistically significant HH clusters
(A2 regions surrounded by A2 neighbours) and 0 LL clusters (A1 regions
surrounded by A1 neighbours). The complete absence of significant LL clusters
is a substantively important finding: catching-up regions are not
geographically cohesive in the same way as advanced innovation regions.
Rather, A1 regions are spatially dispersed — a peripheral fringe surrounding
concentrated A2 cores — consistent with a 'hub-and-periphery' model of
EU regional innovation geography (Crescenzi and Rodríguez-Pose, 2011).
121 regions show no significant local spatial association (NS),
reflecting the interior diversity of the NUTS2 landscape.

## 5.4 Country Fixed-Effects Decomposition

After country-demeaning, all four dimension scores retain large
positive effects (d ≥ 0.67). The retained proportion ranges from
35% (prosperity score,
d_within = +0.822)
to 45% (talent score,
d_within = +0.962).

| Dimension | d (raw) | d (within-country) | % retained |
|-----------|---------|-------------------|-----------|
| Prosperity | +2.347 | +0.822 | 35% |
| Talent | +2.127 | +0.962 | 45% |
| Digital infra. | +1.869 | +0.668 | 36% |
| Innovation cluster | +1.998 | +0.804 | 40% |

Knowledge economy specialisation features retain the highest within-country
signal: LQ KIS hi-tech (63%), LQ ICT services (63%), and tertiary attainment
(55%) retain over half their gross effect after demeaning, suggesting these
reflect genuine regional specialisation rather than purely country-level
structural differences. GDP per capita and R&D metrics, by contrast, are
more strongly mediated by the national context (32–43% retained).

The demeaned logistic regression achieves 70.7% accuracy (versus 99.2%
without demeaning), confirming that approximately 30 percentage points of
discriminative accuracy is attributable to country-level effects. 14
of 27 member states contain regions in both archetypes (mixed countries),
providing direct within-country comparisons for the remaining 70.7%
classification accuracy.

## 5.5 Sectoral Co-Specialisation Typology

Of the 242 regions, 190
(79%)
show specialisation in at least one of the four sector dimensions (LQ > 1.0).
95
regions are co-specialised in two or more dimensions.

The dominant single-sector pattern is **Energy/utilities only** (n = 78),
which is strongly associated with catching-up status
(A2 share = 26%,
transition share = 58%).
Over-representation in energy production without co-specialisation in
innovation-relevant sectors appears to be a structural characteristic
of peripheral catching-up economies, consistent with resource-based
regional economies in the Visegrád group and Mediterranean periphery.

The **Diversified innovation** pattern (J, K, and C all > 1.0; n = 33)
is predominantly associated with A2 membership
(A2 share = 91%).
These regions show mean GDP per capita of €51,915 PPS,
138% of the EU27 mean,
confirming their status as the structural innovation core of the EU.

The **Energy-anchored mixed** pattern (n = 36)
splits evenly between archetypes (A2 share = 50%), with a high transition
region share (44%).
These regions combine energy sector over-representation with at least one
innovation-relevant specialisation, suggesting they are structurally mid-transition.

# 6. Discussion

## 6.1 Interpretation of the Two-Archetype Solution

The k = 2 solution captures the most robust structural divide in EU regional
innovation capacity. The clean separation on prosperity, talent, and
innovation dimensions, combined with the strong spatial autocorrelation
in A2 membership, suggests this is not a statistical artefact but a
genuine structural boundary in the EU regional economy.

The absence of LL spatial clusters for A1 is theoretically important.
It implies that catching-up regions do not form contiguous zones of
low-innovation geography — they are dispersed relative to the concentrated
A2 core. Policy frameworks that treat the catching-up periphery as a
geographically coherent zone may therefore be misspecified. Catching-up
regions in France, Germany, and Spain may face structurally different
challenges than catching-up regions in Poland or Greece, even though
they share the same archetype label.

## 6.2 Within-Country vs. Between-Country Heterogeneity

The country fixed-effects decomposition reveals a nuanced picture.
Approximately 35–45% of the gross archetype separation in dimension scores
reflects genuine within-country regional differences. This is substantively
significant: it means that Germany contains both A1 and A2 regions,
that France's NUTS2 landscape spans the full spectrum, and that Italy's
north-south divide is captured by the archetype taxonomy rather than being
obscured by country-level averaging. The 14 mixed-country member
states provide the empirical basis for arguing that the taxonomy
characterises regional heterogeneity, not merely national heterogeneity.

## 6.3 Limitations

**Cross-sectional design.** All data are for a single reference period.
The taxonomy characterises the structural snapshot circa 2019–2022 and
cannot speak to whether regions are converging or diverging, nor to the
speed at which archetype transitions occur.

**K = 2 simplification.** The k = 3 solution (silhouette = 0.338,
marginally below threshold) would reveal a more differentiated mid-tier
that the current solution assigns to A2. The within-A2 heterogeneity
(e.g., CV > 0.50 for EPO patents and BERD) signals this limitation.
Sensitivity analysis at k = 3 is reported in Appendix A.

**Island regions.** 21 NUTS2 island units have no Queen contiguity
neighbours and are excluded from spatial statistics. This includes
Malta, Cyprus, and several archipelago regions.

**Missing features.** Average wages, Horizon Europe participation,
startup density, and renewable energy generation could not be sourced
at NUTS2 resolution. Their absence may affect score construction,
particularly the innovation cluster dimension.

**Imputation.** Median imputation for MNAR R&D values may understate
the innovation gap for structurally zero-R&D regions. Results for
R&D-related features should be interpreted with this in mind.

# 7. Conclusion

This paper presents a data-driven descriptive taxonomy of EU NUTS2 regional
innovation ecosystems. The k = 2 archetype solution identifies a structural
divide between 134 advanced innovation regions — concentrated in northern
and western Europe, above-average on all four dimension scores, and forming
spatially coherent clusters — and 108 catching-up and peripheral regions,
structurally below-average on all dimensions, dispersed geographically, and
predominantly over-specialised in energy/utilities rather than innovation-
intensive sectors. The taxonomy is robust to alternative clustering methods
(Ward ARI = 0.615) and retains meaningful discriminative power after
country fixed-effects removal (within-country d ≥ 0.67 for all dimensions).

The spatial analysis establishes that advanced innovation regions are not
merely nationally-defined units — they form spatially coherent clusters
(82 HH LISA units) while catching-up regions are structurally dispersed.
The co-specialisation typology adds a sectoral dimension to this picture:
diversified innovation co-specialisation (ICT + KIS + hi-tech manufacturing)
is almost exclusively associated with A2 membership, while energy-sector
dominance without innovation co-specialisation is the modal profile of A1.

These findings are observational and descriptive. They characterise the
structural landscape of EU regional innovation circa 2019–2022 and provide
an empirical foundation for future longitudinal and causal analyses.

# References

- Anselin, L. (1995). Local indicators of spatial association — LISA. *Geographical Analysis*, 27(2), 93–115.
- Asheim, B. T., & Cooke, P. (1999). Local learning and interactive innovation networks in a global economy. In E. Malecki & P. Oinas (Eds.), *Making Connections*. Ashgate.
- Audretsch, D. B., & Feldman, M. P. (1996). R&D spillovers and the geography of innovation and production. *American Economic Review*, 86(3), 630–640.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Cooke, P. (1992). Regional innovation systems: competitive regulation in the new Europe. *Geoforum*, 23(3), 365–382.
- Crescenzi, R., & Rodríguez-Pose, A. (2011). Reconciling top-down and bottom-up development policies. *Environment and Planning A*, 43(4), 773–780.
- Eurostat (2023a). *Regional Innovation Statistics*. European Commission. https://ec.europa.eu/eurostat
- Eurostat (2023b). *R&D Expenditure in All Sectors by NUTS 2 Regions* [rd_e_gerdreg]. European Commission.
- Eurostat (2023c). *Human Resources in Science and Technology by NUTS 2* [hrst_st_rcat]. European Commission.
- Eurostat (2023d). *Patent Applications to the EPO by Priority Year by NUTS 3 Regions* [pat_ep_rtot]. European Commission.
- Eurostat (2023e). *Employment in Technology and Knowledge-Intensive Sectors by NUTS 2* [htec_emp_reg2]. European Commission.
- Eurostat (2023f). *Broadband Internet Coverage by Urban-Rural Typology* [isoc_r_broad_h]. European Commission.
- Feldman, M. P. (1994). *The Geography of Innovation*. Springer.
- Hollanders, H., Es-Sadki, N., & Khalilova, A. (2023). *European Regional Innovation Scoreboard 2023*. European Commission, Directorate-General for Internal Market, Industry, Entrepreneurship and SMEs.
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17–23.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.

# Appendix A: Sensitivity Analysis — k = 3 Cluster Solution

The k = 3 solution (silhouette = 0.338, marginally below the 0.35 threshold)
splits A2 into two sub-groups. Ward ARI improves substantially to 0.815 at
k = 3 (vs 0.615 at k = 2), indicating stronger cross-method agreement.

| Cluster | n | Transition | Mean normalised scores | Countries |
|---------|---|-----------|----------------------|----------|
| C1 Peripheral | 78 | 67 (86%) | cost=0.26 / talent=0.28 / infra=0.38 / cluster=0.28 | GR, PL, IT |
| C2 Innovation leaders | 60 | 0 (0%) | cost=0.68 / talent=0.69 / infra=0.76 / cluster=0.76 | DE, NL, BE |
| C3 Mid-tier | 104 | 19 (18%) | cost=0.51 / talent=0.49 / infra=0.56 / cluster=0.50 | DE, FR, ES |

The k = 3 solution reveals that the A2 archetype in the k = 2 solution
conflates NW European innovation leaders (C2: NL, BE, high-GERD German
regions) with mid-tier performers (C3: France, Spain, remaining German regions).
The primary finding — structural divergence between catching-up periphery
and advanced innovation economies — is robust to this alternative solution;
C1 at k = 3 is essentially a subset of A1 at k = 2.

# Appendix B: Dimension Score Construction Formulas

Let $x_i$ denote the value of feature $x$ for region $i$, $\bar{x}$ the
EU27 sample mean, and $s_x$ the sample standard deviation.

$$\text{score\_cost}_i = \frac{\log_1(\text{GDP\_PPS}_i) - \overline{\log_1(\text{GDP\_PPS})}}
{s_{\log_1(\text{GDP\_PPS})}}$$

$$\text{score\_talent}_i = \text{PC1}[\text{HRST}, \text{Tertiary}, \text{Employment}]_i$$

$$\text{score\_infra}_i = \frac{1}{2}\left(z(\text{Broadband})_i + z(\text{EnternetUse})_i\right)$$

$$\text{score\_cluster}_i = \text{PC1}[\text{GERD, BERD, EPO, GVA\_ICT, LQ\_J, LQ\_K, LQ\_C, LQ\_D}]_i$$

where PC1 denotes the first principal component (scikit-learn `PCA`,
`random_state=42`), fitted on the standardised feature matrix,
and $z(\cdot)$ denotes z-score standardisation.