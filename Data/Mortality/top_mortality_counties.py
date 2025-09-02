### 4/11/25, EB: Talked to Dr. Maroulas yesterday, after meeting with Andrew. He told me we need to not complicate things first, and just look at the data.
### So before I return to the classification or regression model, here I will be looking at the top mortality rates, and seeing which counties are most persistent.
### Now that I say that, I guess this is just the hotspots over time? I'll be looking at the top 5% of counties, and seeing how many times they appear in the top 5% over the years.

# import pandas as pd
# import numpy as np

# # Load the mortality data
# df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
# df['FIPS'] = df['FIPS'].str.zfill(5)

# # List of years to analyze
# year_cols = [col for col in df.columns if col.endswith('MR')]
# years = [int(col.split()[0]) for col in year_cols]

# # Store top 5% FIPS per year
# top_10_fips_by_year = {}

# for year_col in year_cols:
#     year = int(year_col.split()[0])
    
#     # Drop rows with missing or zero mortality values (optional)
#     year_df = df[['FIPS', year_col]].dropna()
    
#     # Determine number of counties to keep (top 5%)
#     n_top = int(np.ceil(len(year_df) * 0.1))

#     # Get top 5% counties
#     top_fips = year_df.sort_values(by=year_col, ascending=False).head(n_top)['FIPS'].tolist()
#     top_10_fips_by_year[year] = set(top_fips)

# # Find intersection: counties present in top 5% every year
# persistent_top_5 = set.intersection(*top_10_fips_by_year.values())

# print(f"✅ Found {len(persistent_top_5)} counties in the top 10% for ALL years.")
# print(sorted(persistent_top_5))


### The above does a pure set intersection, to get ONLY the counties that appear in all years.
### I want to see the counties that appear in the top 5% for at least 10 years. So I will modify the code to count the number of appearances.

# import pandas as pd
# import numpy as np
# from collections import Counter

# # Load the mortality data
# df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
# df['FIPS'] = df['FIPS'].str.zfill(5)

# # List of years to analyze
# year_cols = [col for col in df.columns if col.endswith('MR')]
# years = [int(col.split()[0]) for col in year_cols]

# # Track frequency of top 10% appearances per FIPS
# fips_counter = Counter()

# for year_col in year_cols:
#     year = int(year_col.split()[0])
    
#     # Drop rows with missing or zero mortality values (optional)
#     year_df = df[['FIPS', year_col]].dropna()
    
#     # Determine number of counties to keep (top 10%)
#     n_top = int(np.ceil(len(year_df) * 0.10))  # Change back to 0.05 if needed

#     # Get top 10% counties
#     top_fips = year_df.sort_values(by=year_col, ascending=False).head(n_top)['FIPS'].tolist()
    
#     # Count each FIPS appearance
#     fips_counter.update(top_fips)

# # Filter counties that appear in top 10% ≥ 10 times
# persistent_top_10_fips = [fips for fips, count in fips_counter.items() if count >= 10]

# # Sort if desired
# persistent_top_10_fips.sort()

# df = pd.DataFrame(persistent_top_10_fips, columns=['FIPS'])
# df['FIPS'] = df['FIPS'].str.zfill(5)

# print(f"✅ Found {len(persistent_top_10_fips)} counties that appeared in the top 10% for ≥10 years.")
# #print(df)#persistent_top_10_fips)

# df.to_csv('Data\Mortality\Final Files\Mortality_top10_percent_counties_10yrs.csv', index=False)


#####4/17/25, EB: This does the same as the above section, but utilizes the log-normal distribution to determine the top 5% of counties over time.
##### I think these should be called the persistently at-risk counties, or something like that.

import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import lognorm

# Load mortality data
df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
df['FIPS'] = df['FIPS'].str.zfill(5)

# List of years
year_cols = [col for col in df.columns if col.endswith('MR')]
years = [int(col.split()[0]) for col in year_cols]

# Count top 10% appearances
fips_counter = Counter()

for year_col in year_cols:
    year = int(year_col.split()[0])
    year_df = df[['FIPS', year_col]].dropna()
    
    # Remove zero values (lognorm can't handle zeros)
    year_df = year_df[year_df[year_col] > 0]

    values = year_df[year_col].values

    # Fit log-normal: allow shape, loc, scale to vary
    shape, loc, scale = lognorm.fit(values)

    # Compute 90th percentile threshold
    cutoff = lognorm.ppf(0.9, shape, loc=loc, scale=scale)

    # Filter counties in top 10% of distribution
    top_fips = year_df[year_df[year_col] > cutoff]['FIPS'].tolist()
    fips_counter.update(top_fips)

# Keep counties appearing in top 10% ≥ 10 times
persistent_top_10_fips = [fips for fips, count in fips_counter.items() if count >= 10]
persistent_top_10_fips.sort()

# Save
out_df = pd.DataFrame(persistent_top_10_fips, columns=['FIPS'])
out_df['FIPS'] = out_df['FIPS'].str.zfill(5)

print(f"✅ Found {len(persistent_top_10_fips)} counties in top 10% (lognorm) for ≥10 years.")
out_df.to_csv('Data/Mortality/Final Files/Mortality_top10_percent_counties_10yrs_lognormal.csv', index=False)



















#############################################################################################################################################################################
### Trying a heatmap approach

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load data
# df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
# df['FIPS'] = df['FIPS'].str.zfill(5)
# year_cols = [col for col in df.columns if col.endswith('MR')]

# # Initialize binary inclusion matrix
# inclusion_matrix = pd.DataFrame(index=df['FIPS'].unique())

# for col in year_cols:
#     year = col.split()[0]
#     year_data = df[['FIPS', col]].dropna()
#     n_top = int(np.ceil(len(year_data) * 0.01))
    
#     top_fips = year_data.sort_values(by=col, ascending=False).head(n_top)['FIPS'].tolist()
#     inclusion_matrix[year] = inclusion_matrix.index.isin(top_fips).astype(int)

# # Keep only counties that appear in top 5% at least once
# inclusion_matrix = inclusion_matrix.loc[inclusion_matrix.sum(axis=1) > 0]

# # Plot as heatmap
# plt.figure(figsize=(12, max(6, len(inclusion_matrix) * 0.25)))
# sns.heatmap(inclusion_matrix, cmap='Greens', cbar=False, linewidths=0.5, linecolor='gray')
# plt.title("Top 1% Mortality County Inclusion (2010–2022)")
# plt.xlabel("Year")
# plt.ylabel("County FIPS")
# plt.tight_layout()
# plt.show()
