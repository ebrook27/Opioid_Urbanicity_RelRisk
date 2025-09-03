# Opioid Urbanicity and Relative Risk Project

This project analyzes opioid overdose mortality rates, focusing on a county's relative risk score/level, rather than raw mortality rates. Additionally, we utilize county urbanicity-level as both a predictor variable and a lens to analyze results after the fact.  
The urbanicity level is defined using the NCHS 6-category classification scheme, built principally around county population size (with a few caveats). We aim to improve predictions from prior state-of-the-art and enrich the analysis further. The majority of my contribution is found in /EB_Urbanicity. Within this are different model types, different plots showcasing a variety of results, and some miscellaneous scripts. 

## Data Sources
### Social Vulnerability Index (SVI)
- Source: [CDC/ATSDR SVI Data](https://www.atsdr.cdc.gov/placeandhealth/svi/index.html)  
- Yearly data from **2010–2022** at the county level.  
- Contains demographic and socioeconomic indicators such as:
  - Below poverty
  - Unemployment
  - No high school diploma
  - Minority status
  - Housing & transportation factors

### Urbanicity Levels
- Source: [NCHS Urban–Rural Classification Scheme](https://www.cdc.gov/nchs/data-analysis-tools/urban-rural.html)  
- Each county is assigned one of **6 categories**, e.g.:
  1. Large Central Metro  
  2. Large Fringe Metro  
  3. Medium Metro  
  4. Small Metro  
  5. Micropolitan  
  6. Non-Core  

For some analyses, we collapse categories into broader groupings (urban vs rural, or 3-category tiers).

---

### Mortality & Relative Risk
- Mortality rates: county-level opioid overdose deaths per 100,000 residents.  
- Relative Risk Score (RRS): a transformation used to highlight **high-risk strata** across counties.  
  - Defined as:  
    \[
    RR = \frac{\text{# cases in top risk strata / # counties in top strata}}
             {\text{# cases in population / # counties in total population}}
    \]
  - This allows us to compare risk concentration across county groupings (e.g., by urbanicity).
