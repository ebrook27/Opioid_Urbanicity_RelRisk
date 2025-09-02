### 4/17/25, EB: Following the advice of VM, I determined which counties appear in the top 10% of mortality for at least 10 of the 13 years we have data for.
# I then took the SVI data for these counties, and clustered them using KMeans. I then applied the urbanicity labels to these counties, and here I will run an ANOVA on the clusters.

import pandas as pd
from functools import reduce
from statsmodels.formula.api import ols
import statsmodels.api as sm


DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

def prepare_anova_input_data():
    svi_variables = [v for v in DATA if v != 'Mortality']

    # Load urban-rural classification
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.rename(columns={'2023 Code': 'Urban_Rural_Code'})
    nchs_df['Urbanicity'] = nchs_df['Urban_Rural_Code'].map({1: 'Urban', 6: 'Rural'})

    # Load persistent counties
    persistent_df = pd.read_csv('County Classification\Persistent_Counties_Cluster_Results.csv', dtype={'FIPS': str})
    persistent_df['FIPS'] = persistent_df['FIPS'].str.zfill(5)

    # Keep only persistent counties with valid urbanicity (1 or 6)
    merged_meta = persistent_df.merge(nchs_df[['FIPS', 'Urbanicity']], on='FIPS')
    merged_meta = merged_meta[merged_meta['Urbanicity'].isin(['Urban', 'Rural'])]

    # Load and merge all SVI variables
    svi_data = []
    for var in svi_variables:
        var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
        var_df = pd.read_csv(var_path, dtype={'FIPS': str})
        var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
        long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)
        long_df['year'] = long_df['year_var'].str.extract(r'(\d{4})').astype(int)
        long_df = long_df[long_df['year'].between(2010, 2021)]
        long_df = long_df.drop(columns='year_var')
        svi_data.append(long_df)

    # Merge on FIPS + year
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Merge with metadata to get Urban/Rural class
    final_df = svi_merged.merge(merged_meta[['FIPS', 'Urbanicity']], on='FIPS', how='inner')
    final_df = final_df.dropna()

    return final_df

def run_anova_on_svi(anova_df):
    anova_results = []

    for var in [v for v in DATA if v != 'Mortality']:
        model = ols(f'Q("{var}") ~ C(Urbanicity)', data=anova_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_table['Variable'] = var
        anova_results.append(anova_table)

    combined = pd.concat(anova_results)
    combined.reset_index(inplace=True)
    summary = combined[['Variable', 'F', 'PR(>F)']].rename(columns={'F': 'F_statistic', 'PR(>F)': 'p_value'})
    summary.sort_values('p_value', inplace=True)

    return summary


def main():
    
    anova_input = prepare_anova_input_data()
    anova_summary = run_anova_on_svi(anova_input)
    print(anova_summary)


if __name__ == "__main__":
    main()