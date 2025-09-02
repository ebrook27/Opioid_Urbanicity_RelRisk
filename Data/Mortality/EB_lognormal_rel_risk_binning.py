### 4/14/25, EB: In the file morality_distribution_fitting.py, I tested several distributions to see which one fit the mortality data best.
### Using the Kolmogorov-Smirnov test, I found that the log-normal distribution fit the data best every year, same as AD found.
### I will use this distribution to bin the mortality data into 20 equally sized bins, and then save the results.
# import pandas as pd
# import numpy as np
# from scipy.stats import lognorm

# # Load mortality data
# df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
# df['FIPS'] = df['FIPS'].str.zfill(5)

# # Year columns to analyze
# year_cols = [col for col in df.columns if col.endswith('MR')]
# percentiles = np.linspace(0, 1, 21)

# # Output DataFrame
# output_df = pd.DataFrame()
# output_df['FIPS'] = df['FIPS']

# for col in year_cols:
#     year = col.split()[0]
#     rates = df[col].copy()
    
#     # Filter for fitting (remove zeros and NaNs)
#     fit_values = rates[(rates > 0) & (~rates.isna())].values
#     if len(fit_values) == 0:
#         print(f"⚠️ Skipping {year} — no valid data.")
#         continue

#     # Fit log-normal
#     shape, loc, scale = lognorm.fit(fit_values)

#     # Compute bin edges and assign bins
#     bin_edges = lognorm.ppf(percentiles, shape, loc=loc, scale=scale)
#     bin_edges[0] = -np.inf
#     bin_edges[-1] = np.inf
#     bin_labels = pd.cut(rates, bins=bin_edges, labels=False)

#     # Add bin label column
#     output_df[f'{year}_RR_Level'] = bin_labels

#     # --- Compute RR per bin ---
#     rr_per_bin = {}
#     total_cases = rates.sum()
#     total_counties = rates.count()

#     for b in range(20):
#         bin_mask = bin_labels == b
#         bin_cases = rates[bin_mask].sum()
#         bin_counties = bin_mask.sum()

#         if bin_counties == 0 or total_cases == 0:
#             rr = 0
#         else:
#             rr = (bin_cases / bin_counties) / (total_cases / total_counties)

#         rr_per_bin[b] = rr

#     # Map RR scores to each county based on their bin
#     rr_scores = bin_labels.map(rr_per_bin)
#     output_df[f'{year}_RR_Score'] = rr_scores

# print("✅ Binning and RR scoring complete for all years.")
# print(output_df.head())

# # Save output
# output_df.to_csv('Data/Mortality/Final Files/Mortality_lognormal_binned_RR.csv', index=False)
# print("💾 Saved to: Mortality_lognormal_binned_RR.csv")


########################################################################################################################################
### 4/21/25, EB: The above bins the counties into 20 equally sized bins based on the log-normal distribution of mortality rates.
### What I've been thinking about, after talking with AS and AD on 4/17/25, is that we are more interested in the highest risk counties, and AS
### said "Look at top 0.1% of risk, 0.5%, 1%, so on...". So what I'm trying to do is to fit a log-normal distribution to the mortality rates,
### and then we can use the distribution to compute each county's percentile rank based on that risk distribution. That can easily be converted into
### relative risk score, simply take that county and all above it as the bin, and compute risk from there.
### Then we can use a regression model to predict the percentile rank based on the SVI variables?

import pandas as pd
import numpy as np
from scipy.stats import lognorm

# Load mortality data
df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
df['FIPS'] = df['FIPS'].str.zfill(5)

# Output list
percentile_rows = []

# Loop over years
year_cols = [col for col in df.columns if col.endswith('MR')]
for col in year_cols:
    year = int(col.split()[0])
    year_df = df[['FIPS', col]].copy().dropna()
    year_df = year_df[year_df[col] > 0]  # Remove zeros
    
    # Fit log-normal (forcing loc=0 for interpretability)
    shape, loc, scale = lognorm.fit(year_df[col], floc=0)
    
    # Compute percentiles using log-normal CDF
    year_df['Mortality'] = year_df[col]
    year_df['Percentile'] = lognorm.cdf(year_df[col], s=shape, loc=loc, scale=scale)
    year_df['Year'] = year
    year_df = year_df[['FIPS', 'Year', 'Mortality', 'Percentile']]
    
    percentile_rows.append(year_df)

# Combine all years
percentile_df = pd.concat(percentile_rows)
# Save output
percentile_df.to_csv('Data/Mortality/Final Files/Mortality_lognormal_percentile_RR.csv', index=False)
print("💾 Saved to: Mortality_lognormal_percentile_RR.csv")