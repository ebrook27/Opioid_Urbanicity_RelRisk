### 4/8/25, EB: This file is to be used in conjunction with the /County Classification/rel_risk_classification.py file.
### Here I am taking the Final Mortality Data and transforming it into a relative risk score.
### The formula for that is:
### Relative risk = (# cases in top risk strata / # samples in top risk strata) / (# cases in population) / (# samples in total population)
### We will perform this calculation for all years, so that we can predict this value rather than the mortality rate.
#####################################################################################
### 4/10/25, EB: The commented-out code below is what I used to create even bins for the classification problem.
### The bit below it is my attempt to create custom bins based on the percentiles of the data. Since we are ultimately
### interested in the top 0.5%, top 1%, top 2%, etc., which are overlapping percentiles, we had to be clever and split
### the percentiles into unique bins. For example, the first bin are the top 0.5% worst counties, the second bin
### are the next 0.5% worst counties, the third is the next 1% worst counties, and so on. This way, we have mutually exclusive
### bins. After the classification, we'll have to re-combine the bins to get the desired interepretation of the data.


# import pandas as pd
# import numpy as np
# import os

# # Paths
# input_path = r'Data\Mortality\Final Files\Mortality_final_rates.csv'
# output_path = r'Data\Mortality\Final Files\Mortality_relative_risk_scores.csv'

# # Load mortality data
# df = pd.read_csv(input_path, dtype={'FIPS': str})
# df['FIPS'] = df['FIPS'].str.zfill(5)

# year_cols = [col for col in df.columns if col.endswith('MR')]

# # Output dataframe
# rr_df = pd.DataFrame()
# rr_df['FIPS'] = df['FIPS']

# # Function to compute RR scores for bins
# def compute_rr_scores(values, n_bins=20):
#     """
#     Bins values into quantiles, then computes relative risk score per bin.
#     Returns RR score for each entry in values.
#     """
#     total_cases = values.sum()
#     total_samples = len(values)

#     if total_cases == 0:
#         return np.zeros_like(values)

#     # Bin values (quantile-based)
#     try:
#         bin_labels = pd.qcut(values, q=n_bins, labels=False, duplicates='drop')
#     except ValueError:
#         # Fallback in case of non-unique values
#         bin_labels = pd.cut(values, bins=n_bins, labels=False)

#     # Compute RR score per bin
#     bin_rr_scores = {}
#     for b in np.unique(bin_labels[~pd.isna(bin_labels)]):
#         in_bin = (bin_labels == b)
#         bin_cases = values[in_bin].sum()
#         bin_count = in_bin.sum()
#         rr = (bin_cases / bin_count) / (total_cases / total_samples)
#         bin_rr_scores[b] = rr

#     # Assign RR score to each row
#     rr_scores = [bin_rr_scores.get(lbl, 0) for lbl in bin_labels]
#     return rr_scores

# # Apply year by year
# for col in year_cols:
#     year = col.split()[0]
#     rr_scores = compute_rr_scores(df[col].values, n_bins=20)
#     rr_df[f'{year} RR_Score'] = rr_scores

# # Save to CSV
# rr_df.to_csv(output_path, index=False)
# #print(rr_df.head())
# print(f"✅ Saved RR-score transformed data to: {output_path}")
#################################################################################################################
### 4/9/25, EB: I realized that the above code was not going to work with the classification script, rel_risk_classification.py.
### Instead of using the relative risk scores as the bin labels, I just need to use the bin labels themselves. The following should
### accomplish that. I am going to keep the above code in case I need to go back to it, but I will comment it out.

import pandas as pd
import numpy as np
import os

# Path to the mortality file
input_path = r'Data\Mortality\Final Files\Mortality_final_rates.csv'
output_path = r'Data\Mortality\Final Files\Mortality_relative_risk_custom_levels.csv'

# Load mortality data
df = pd.read_csv(input_path, dtype={'FIPS': str})
df['FIPS'] = df['FIPS'].str.zfill(5)

# Extract all year columns like "2010 MR", "2011 MR", ...
year_cols = [col for col in df.columns if col.endswith('MR')]

# Create a new DataFrame to hold FIPS and RR level columns
rr_df = pd.DataFrame()
rr_df['FIPS'] = df['FIPS']

# Define custom percentile thresholds
custom_percentiles = [
    0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0
]

# def compute_rr_bins(values, percentiles=custom_percentiles):
#     """
#     Rank counties and assign 20 quantile-based bins.
#     Guarantees all bins are filled, even with duplicate mortality values.
#     """
#     total = values.sum()
#     if total == 0:
#         return np.zeros_like(values)

#     # Normalize to relative risk
#     rr = values / total

#     # Convert percentiles to RR thresholds
#     unique_thresholds = np.unique(np.quantile(rr, percentiles))

#     # Bin values using np.digitize (returns bin indices from 1 to len(thresholds))
#     bins = np.digitize(rr, bins=unique_thresholds, right=True)  # right=True makes bin edges inclusive on the right

#     ### 4/9/25, EB: The following commented code is for even sized bins. I'm
#     ### going to use the custom_percentiles instead.
#     # # Rank values to avoid duplicate cutoffs
#     # ranks = pd.Series(rr).rank(method='first')
    
#     # # Bin ranks into 20 quantiles (0 = lowest, 19 = highest)
#     # bins = pd.qcut(ranks, q=20, labels=False)
    
#     return bins

def compute_rr_bins(values, percentiles):
    """
    Bin counties using reversed relative risk percentiles, so bin 0 = highest risk.
    """
    total = values.sum()
    if total == 0:
        return np.zeros_like(values)

    rr = values / total

    # Reversed percentile rank: 0 = highest RR, 1 = lowest RR
    percent_rank = pd.Series(rr).rank(method='first', ascending=False) / len(rr)

    # Use pd.cut to bin based on reversed percentile rank
    bins = pd.cut(percent_rank, bins=[0.0] + percentiles, labels=False, include_lowest=True)

    return bins


# Apply to each year column
for col in year_cols:
    rr_bins = compute_rr_bins(df[col].values, custom_percentiles)
    year = col.split()[0]
    rr_df[f'{year} RR_Level'] = rr_bins

# Save to CSV
rr_df.to_csv(output_path, index=False)
print(f"✅ Saved RR-level binned data to: {output_path}")
