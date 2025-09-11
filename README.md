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
    ```math
    RR = \frac{\text{(No. of cases in top risk strata) / (No. of counties in top strata)}}
             {\text{(No. of cases in population) / (No. of counties in total population)}}
    ```
  - This allows us to compare risk concentration across county groupings (e.g., by urbanicity).

### TODO:  
- [ ] [Custom loss for XGBoost, weighting high risk bins/counties more](https://pubmed.ncbi.nlm.nih.gov/39277561/)  
- [ ] Predict mortality, sort zero-mortality counties into own bin, fit Log-Normal to rest. Compare to actual log-normal  
- [ ] [Tweedie XGBoost?](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters) (this made the mortality predictions fit the zero-inflated distribution better, but is that necessarily what we want?) <- It seems like this didn't improve MR predictions but, again, what's the goal?  
- [ ] Weighted %BPR for each risk category, to use as loss function?
- [ ] Do we want to consider top 10% of ALL counties or top 10% without counting the zero counties? (Probably both and compare)
- [ ] How do we visualize each county's mortality rate over time (i.e. for a particular county, how does it's mortality rate compare year to year? What distribution does that take?)
- [ ] How do all of the variables actually relate? How do the different models perform with these to predict what we want? (Advantages, limitations...)
