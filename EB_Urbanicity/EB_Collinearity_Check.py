### 8/13/25, EB: Did the oral exam yesterday, and there were several good questions raised. The most straightforward one was about collinearity.
### I have not checked for collinearity among the SVI data and the urbanicity classes, but there is certainly some.
### For example, how many mobile homes are there in NYC? Surely 0. So this script is to check for collinearity among the SVI variables and the urbanicity classes.

### 8/14/25, EB: Alright, realized I made a big oversight error. The datasets I'm using are percentile ranks, standardized to be between 0 and 100.
### Because of this,  

import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from scipy.stats import f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

### This one prepares the mortality rates directly.
def prepare_yearly_prediction_data_mortality():
    """
    Prepares a long-format dataset for predicting next-year Mortality Rate
    using current-year SVI + county class as inputs.
    """
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2023))

    # Load county urbanicity class
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

    # Load mortality data
    mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
    mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
    mort_df = mort_df.set_index('FIPS')

    # Load and reshape SVI variables
    svi_data = []
    for var in svi_variables:
        var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
        var_df = pd.read_csv(var_path, dtype={'FIPS': str})
        var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
        long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)
        long_df['year'] = long_df['year_var'].str.extract(r'(\d{4})').astype(int)
        long_df = long_df[long_df['year'].between(2010, 2022)]
        long_df = long_df.drop(columns='year_var')
        svi_data.append(long_df)

    # Merge all SVI variables into one long dataframe
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Add next-year Mortality Rate (no log transform)
    for y in years:
        mr_col = f'{y+1} MR'
        if mr_col not in mort_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'mortality_next'] = svi_merged.loc[
            svi_merged['year'] == y, 'FIPS'].map(mort_df[mr_col])

    # Add urbanicity class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows with missing data
    svi_merged = svi_merged.dropna()

    return svi_merged, svi_variables


def plot_mean_heatmap(mean_df):
    """
    Plots a heatmap of mean values for SVI variables across urbanicity classes.
    """
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        mean_df,
        annot=False,
        fmt=".2f",
        cmap="vlag",
        cbar_kws={'label': 'Mean SVI Value'},
        linewidths=0.5,
        linecolor='gray'
    )

    # Manual annotation
    for i in range(mean_df.shape[0]):
        for j in range(mean_df.shape[1]):
            val = mean_df.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                    ha='center', va='center', color='black', fontsize=9)

    ax.set_title("Mean SVI Variable Values by Urbanicity Class", fontsize=14)
    ax.set_ylabel("SVI Variable", fontsize=12)
    ax.set_xlabel("Urbanicity Class", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("mean_svi_by_urb_category_manual.png", dpi=300)
    plt.show()


def means_corr_plots(svi_merged, svi_vars):
    """ Plot means of SVI variables by urbanicity class and calculate correlations.
    """

    urban_group_means = svi_merged.groupby('county_class')[svi_vars].mean().T

    # plt.figure(figsize=(12, 8))
    # sns.heatmap(urban_group_means, cmap='vlag', annot=True, fmt=".2f", cbar_kws={'label': 'Mean Value'})
    # plt.title("Mean SVI Variable Values by Urbanicity Class")
    # plt.xlabel("Urbanicity Class")
    # plt.ylabel("SVI Variable")
    # plt.tight_layout()
    # plt.show()
    plot_mean_heatmap(urban_group_means)


def point_biserial_heatmap(corr_df):
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        corr_df,
        annot=False,  # turn off default annotations
        fmt=".2f",
        center=0,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Correlation'}
    )

    # Add manual annotations
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            val = corr_df.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                    ha='center', va='center', color='black', fontsize=9)

    ax.set_title("Point-Biserial Correlation: Urbanicity Class vs SVI Variables", fontsize=14)
    ax.set_ylabel("SVI Variable", fontsize=12)
    ax.set_xlabel("Urbanicity Class", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("manual_annotation_heatmap.png", dpi=300)
    plt.show()


def point_biserial_correlation(svi_merged, svi_vars):
    """
    Calculate point-biserial correlation between SVI variables and urbanicity classes.
    """
    svi_copy = svi_merged.copy()
    urban_classes = sorted(svi_merged['county_class'].unique())
    results = {}

    for uclass in urban_classes:
        svi_copy[f'class_{uclass}'] = (svi_copy['county_class'] == uclass).astype(int)
        corrs = {}
        for var in svi_vars:
            r, _ = pointbiserialr(svi_copy[f'class_{uclass}'], svi_copy[var])
            corrs[var] = r
        results[f'Class {uclass}'] = corrs

    # Convert results to DataFrame for easier handling
    corr_df = pd.DataFrame(results)
    
    # plt.figure(figsize=(10, 8))
    # ### Making the color bar heatmap to go from -1 to 1, to not be misleading.
    # #sns.heatmap(corr_df, annot=True, fmt=".2f", center=0, cmap="coolwarm", cbar_kws={'label': 'Correlation'})
    # sns.heatmap(
    #     corr_df, 
    #     annot=True, 
    #     fmt=".2f", 
    #     center=0, 
    #     cmap="coolwarm", 
    #     vmin=-1, vmax=1,  # Force color scale from -1 to +1
    #     cbar_kws={'label': 'Correlation'},
    #     annot_kws={"color": "black"}
    # )
    # plt.title("Point-Biserial Correlation: Urbanicity Class vs SVI Variables")
    # plt.ylabel("SVI Variable")
    # plt.xlabel("Urbanicity Class")
    # plt.tight_layout()
    # plt.show()
    
    ####################################
    # plt.figure(figsize=(12, 10))

    # sns.heatmap(
    #     corr_df,
    #     annot=True,
    #     fmt=".2f",
    #     center=0,
    #     cmap="coolwarm",
    #     vmin=-1,
    #     vmax=1,
    #     linewidths=0.5,
    #     linecolor='gray',
    #     cbar_kws={'label': 'Correlation'},
    #     annot_kws={"color": "black", "size": 9}
    # )

    # plt.title("Point-Biserial Correlation: Urbanicity Class vs SVI Variables", fontsize=14)
    # plt.ylabel("SVI Variable", fontsize=12)
    # plt.xlabel("Urbanicity Class", fontsize=12)
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=0)
    # plt.tight_layout()

    # # Save instead of show
    # #plt.savefig("County Classification\correlation_heatmap_debug.png", dpi=300)
    # plt.show()

    point_biserial_heatmap(corr_df)
    return corr_df


def anova_test(svi_merged, svi_vars):
    """
    Perform ANOVA test for each SVI variable across urbanicity classes.
    Returns a DataFrame with p-values and significance.
    """
    
    # Perform ANOVA for each SVI variable
    urban_classes = sorted(svi_merged['county_class'].unique())

    anova_results = {}
    for var in svi_vars:
        groups = [svi_merged[svi_merged['county_class'] == uc][var] for uc in urban_classes]
        f_val, p_val = f_oneway(*groups)
        anova_results[var] = p_val

    anova_df = pd.DataFrame.from_dict(anova_results, orient='index', columns=['p-value'])
    anova_df['Significant (p < 0.05)'] = (anova_df['p-value'] < 0.05)
    anova_df['Significance Label'] = anova_df['Significant (p < 0.05)'].map({True: 'Significant', False: 'Not Significant'})
    anova_df['-log10(p-value)'] = -np.log10(anova_df['p-value'])
    
    print(anova_df.sort_values(by='p-value'))

    plt.figure(figsize=(10, 6))
    #sns.barplot(data=anova_df.reset_index(), x='-log10(p-value)', y='index', 
    #            hue='Significant (p < 0.05)', dodge=False)
    sns.barplot(
        data=anova_df.reset_index(),
        x='-log10(p-value)',
        y='index',
        hue='Significance Label',
        dodge=False
    )
    plt.title("ANOVA Test Results: SVI Variables across Urbanicity Classes")
    plt.xlabel("-log10(p-value)")
    plt.ylabel("SVI Variable")
    plt.tight_layout()
    plt.show()

    return anova_df


# def compute_vif_matrix(df, predictor_vars, categorical_var):
#     """
#     Computes VIFs for continuous + categorical predictors.
#     """
#     # Construct formula for design matrix with dummy encoding
#     all_predictors = predictor_vars + [f"C({categorical_var})"]
#     formula = ' + '.join(all_predictors)
#     y, X = dmatrices(f"mortality_next ~ {formula}", data=df, return_type='dataframe')

#     vif_data = pd.DataFrame()
#     vif_data["Variable"] = X.columns
#     vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     return vif_data.sort_values(by='VIF', ascending=False)


def compute_vif_matrix(df, predictor_vars, categorical_var):
    """
    Computes VIFs for continuous + categorical predictors.
    This one cleans the variable names first to be compatible with patsy.
    """
    # Clean column names for patsy
    df = df.rename(columns=lambda x: x.replace(" ", "_").replace("-", "_"))
    predictor_vars = [v.replace(" ", "_").replace("-", "_") for v in predictor_vars]
    categorical_var = categorical_var.replace(" ", "_")

    # Build formula
    all_predictors = predictor_vars + [f"C({categorical_var})"]
    formula = ' + '.join(all_predictors)
    
    # Create design matrices
    y, X = dmatrices(f"mortality_next ~ {formula}", data=df, return_type='dataframe')

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values(by='VIF', ascending=False)




###############################################################
# def main():
#     df, svi_vars = prepare_yearly_prediction_data_mortality()
    
#     df['county_class'] = df['county_class'].astype(int)
#     svi_vars = [v for v in DATA if v != 'Mortality']
#     corrs = df[svi_vars + ['county_class']].corr()

#     # Correlation of each SVI variable with urbanicity
#     urban_corr = corrs['county_class'].drop('county_class').sort_values()

#     # Plotting
#     plt.figure(figsize=(8, 10))
#     urban_corr.plot(kind='barh', title='SVI vs Urbanicity Correlation')
#     plt.axvline(0, color='black', linewidth=0.8)
#     plt.tight_layout()
#     plt.show()

def main():
    # # Prepare data
    # df, svi_vars = prepare_yearly_prediction_data_mortality()
    # # print(df['county_class'].head())
    # # print(df['county_class'].dtype)


    # # Ensure county_class is categorical
    # df['county_class'] = df['county_class'].astype(str)

    # # Run all 3 correlation/collinearity analyses
    # print("📊 Running group means visualization...")
    # means_corr_plots(df, svi_vars)

    # print("\n📈 Running point-biserial correlation...")
    # pb_corr_df = point_biserial_correlation(df, svi_vars)
    # #print(pb_corr_df.dtypes)
    # #print(pb_corr_df)
    
    # # print("\n📉 Running ANOVA test...")
    # # anova_df = anova_test(df, svi_vars)

    ########################
    # VIF Calculation
    # === RUN THIS ===
    df, svi_vars = prepare_yearly_prediction_data_mortality()
    vif_table = compute_vif_matrix(df, svi_vars, categorical_var='county_class')
    print(vif_table)

    
    
if __name__ == '__main__':
    main()
