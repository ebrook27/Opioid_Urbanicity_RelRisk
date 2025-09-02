### 5/12/25, EB: I tried using anomaly detection to find outliers in the SVI data. My (and VM's) hope was that this approach would help us identify
### the high-risk counties solely from the SVI data. It did not work, and instead mostly identified counties with very low risk levels. There were
### some high mortality counties some years, but not consistently, and not with any frequency to really justify the approach. We might be able to
### use the results to help do comparative analysis with the high-risk counties, but that will come later down the line.
### My goal with this script is to find some sort of unsupervised clustering method that can help us identify high-risk counties.
### I hope we can identify a cluster corresponding to high-risk counties, and use the patterns in the SVI data to help us perform some sort of 
### dimension reduction to help us identify the most important features for predicting high-risk counties.
### At the very least, I hope we can maybe identify some risk-level profiles, that we can use in a predictive model to predict risk levels.
### Here I will try a few different clustering methods, and see if any of them can help us identify high-risk counties.

### 5/12/25, EB: I tried several different clustering methods, including KMeans, DBSCAN, Gaussian Mixture Models (GMM), and Hierarchical Clustering.
### I tried several different numbers of clusters, and it seems like k-means and GMM are the most promising, with 4 or 5 clusters resultign in the best
### mortality rate distributions.
### I am now going to look at the SVI "profiles" for each cluster, and see if I can identify any patterns in the SVI data that correspond to high-risk counties.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns


DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

def construct_data_df():
    """Constructs the data_df with full 6-class urban-rural codes."""
    
    # Initialize with Mortality data (same as before)
    mortality_path = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
    data_df = pd.read_csv(
        mortality_path,
        header=0,
        names=['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)],
        dtype={'FIPS': str}
    )
    data_df['FIPS'] = data_df['FIPS'].str.zfill(5)

    # Load other variables (unchanged)
    for variable in [v for v in DATA if v != 'Mortality']:
        variable_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
        
        if variable == 'Social Capital':
            sci_df = pd.read_csv(
                variable_path,
                usecols=['FIPS', '2018 Social Capital'],
                dtype={'FIPS': str}
            )
            sci_df['FIPS'] = sci_df['FIPS'].str.zfill(5)
            data_df = pd.merge(data_df, sci_df, on='FIPS', how='left')
        else:
            var_df = pd.read_csv(
                variable_path,
                header=0,
                names=['FIPS'] + [f'{year} {variable}' for year in range(2010, 2023)],
                dtype={'FIPS': str}
            )
            var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
            data_df = pd.merge(data_df, var_df, on='FIPS', how='left')

    # Load NCHS data with 6-class codes
    urban_rural = pd.read_csv(
        'Data/SVI/NCHS_urban_v_rural.csv',
        dtype={'FIPS': str},
        usecols=['FIPS', '2023 Code']
    )
    urban_rural['FIPS'] = urban_rural['FIPS'].str.zfill(5)

    # Merge and rename target column
    data_df = pd.merge(
        data_df,
        urban_rural,
        on='FIPS',
        how='left'
    ).rename(columns={'2023 Code': 'urban_rural_class'})


    # Convert classes to 0-5 (if originally 1-6)
    data_df['urban_rural_class'] = data_df['urban_rural_class'].astype(int) - 1  # Optional: adjust to 0-based

    # print("Missing class labels:", data_df['urban_rural_class'].isna().sum())
    # # Verify labels are 0-5
    # print("Unique classes:", data_df['urban_rural_class'].unique())
    
    return data_df

def prepare_clustering_data(data_df, year=2020, scale_data=False):
    # Select SVI features for the specified year
    svi_vars = [v for v in DATA if v != 'Mortality']
    year_cols = [f"{year} {v}" for v in svi_vars if f"{year} {v}" in data_df.columns]
    
    features = ['FIPS', f"{year} Mortality Rates"] + year_cols
    df = data_df[features].dropna().copy()
    df = df.rename(columns={f"{year} Mortality Rates": "MortalityRate"})
    
    # Extract SVI feature matrix
    X = df[year_cols].values
    if scale_data:
        X_scaled = StandardScaler().fit_transform(X)

    return df, X_scaled, year_cols

# We'll cluster using three methods: KMeans, DBSCAN, and GMM
def run_clustering_algorithms(df, X_scaled, n_clusters=4):
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    df['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

    # GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    df['Cluster_GMM'] = gmm.fit_predict(X_scaled)
    
    # Hierarchical Clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    df['Cluster_Hierarchical'] = hc.fit_predict(X_scaled)

    return df

# # Analyze mortality distribution by cluster
# def plot_cluster_mortality_distributions(df):
    plt.figure(figsize=(16, 5))

    for i, method in enumerate(['Cluster_KMeans', 'Cluster_DBSCAN', 'Cluster_GMM', 'Cluster_Hierarchical']):
        plt.subplot(1, 3, i+1)
        sns.boxplot(data=df, x=method, y='MortalityRate')
        plt.title(f'{method} vs Mortality')
        plt.xlabel('Cluster Label')
        plt.ylabel('Mortality Rate')

    plt.tight_layout()
    plt.show()

# def plot_cluster_mortality_distributions(df):
    """
    Plot mortality distributions by cluster for each clustering method.
    This one produces a 2x2 grid of plots.
    """
    cluster_methods = [
        ('Cluster_KMeans', 'KMeans'),
        ('Cluster_DBSCAN', 'DBSCAN'),
        ('Cluster_GMM', 'GMM'),
        ('Cluster_Hierarchical', 'Hierarchical')
    ]

    plt.figure(figsize=(14, 10))  # Wider & taller for clarity

    for i, (col, title) in enumerate(cluster_methods, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=df, x=col, y='MortalityRate')
        plt.title(f'{title} Clustering vs Mortality')
        plt.xlabel('Cluster Label')
        plt.ylabel('Mortality Rate')

    plt.tight_layout()
    plt.show()


def plot_cluster_mortality_distributions(df):
    """
    Plot mortality distributions by cluster for each clustering method.
    This one produces a 2x2 grid of plots, and contains a horizontal line
    for the national average mortality rate.
    """
    cluster_methods = [
        ('Cluster_KMeans', 'KMeans'),
        ('Cluster_DBSCAN', 'DBSCAN'),
        ('Cluster_GMM', 'GMM'),
        ('Cluster_Hierarchical', 'Hierarchical')
    ]

    # Compute national average mortality rate
    national_avg = df['MortalityRate'].mean()

    plt.figure(figsize=(14, 10))

    for i, (col, title) in enumerate(cluster_methods, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=df, x=col, y='MortalityRate')
        plt.axhline(national_avg, color='black', linestyle='--', linewidth=1.5)
        plt.title(f'{title} Clustering vs Mortality')
        plt.xlabel('Cluster Label')
        plt.ylabel('Mortality Rate')
        #plt.text(0.95, national_avg + 1, f'Nat. Avg: {national_avg:.1f}',
        #         ha='right', va='bottom', color='black', fontsize=9)

    plt.tight_layout()
    plt.show()

##############################
### Interpreting the clusters:

def profile_kmeans_clusters(df, svi_vars, cluster_col='Cluster_KMeans'):
    """
    Computes average SVI values for each KMeans cluster.

    Args:
        df: DataFrame containing SVI data and cluster labels.
        svi_vars: List of SVI variable names (e.g., ['Below Poverty', 'Unemployment', ...])
        cluster_col: Column with KMeans cluster labels.

    Returns:
        A DataFrame with average SVI values per cluster.
    """
    cluster_profiles = (
        df.groupby(cluster_col)[svi_vars]
        .mean()
        .round(2)
        .sort_index()
    )
    return cluster_profiles

def profile_clusters_by_urbanicity(df, svi_vars, cluster_col='Cluster_KMeans', urban_col='urban_rural_class'):
    """
    Computes average SVI values for each (cluster, urbanicity) combination.

    Args:
        df: DataFrame with SVI values, cluster labels, and urbanicity codes.
        svi_vars: List of SVI variable names.
        cluster_col: Name of cluster label column.
        urban_col: Name of urbanicity column (should be categorical/int 0–5).

    Returns:
        A MultiIndex DataFrame: (cluster, urbanicity) × SVI averages
    """
    grouped = (
        df.groupby([cluster_col, urban_col])[svi_vars]
        .mean()
        .round(2)
    )
    return grouped




###############################
### 5/13/25, EB: So the above code is working, and we see some of what we wanted: using 4 clusters seemed to discriminate based on mortality rate best,
### and we found that two  of the clusters had, overall, higher mortality rates, for the year 2020.
### Now I'm interested in seeing how we can extend this to the other years, and see if we can find any patterns in the SVI data that correspond to high-risk counties.

def reshape_multi_year_data(data_df, svi_vars, years=range(2010, 2023)):
    long_df = []

    for year in years:
        year_cols = [f"{year} {v}" for v in svi_vars if f"{year} {v}" in data_df.columns]
        if not year_cols:
            continue

        temp_df = data_df[['FIPS', f"{year} Mortality Rates"] + year_cols].dropna().copy()
        temp_df['Year'] = year
        temp_df = temp_df.rename(columns={f"{year} Mortality Rates": "MortalityRate",
                                          **{f"{year} {v}": v for v in svi_vars}})
        long_df.append(temp_df)

    return pd.concat(long_df, ignore_index=True)

def cluster_pooled_data(long_df, svi_vars, method='kmeans', n_clusters=4):

    X = long_df[svi_vars].values
    X_scaled = StandardScaler().fit_transform(X)

    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
        long_df['Cluster_KMeans'] = model.fit_predict(X_scaled)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        long_df['Cluster_GMM'] = model.fit_predict(X_scaled)
    else:
        raise ValueError("Unsupported method")

    return long_df

### Year over year clustering functions:
def cluster_and_profile_by_year(data_df, svi_vars, years=range(2010, 2023), n_clusters=4):
    all_years = []

    for year in years:
        year_cols = [f"{year} {v}" for v in svi_vars if f"{year} {v}" in data_df.columns]
        if not year_cols:
            continue

        required_cols = ['FIPS', f"{year} Mortality Rates"] + year_cols
        df_year = data_df[required_cols].dropna().copy()
        df_year['Year'] = year
        df_year = df_year.rename(columns={f"{year} Mortality Rates": "MortalityRate",
                                          **{f"{year} {v}": v for v in svi_vars}})

        X = df_year[svi_vars].values
        ### 5/14/25, EB: The data is already normalized from 0-100, so scaling is not necessary. I tested both ways and found it didn't make a difference.
        #X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        df_year['Cluster_KMeans'] = kmeans.fit_predict(X)#_scaled)

        all_years.append(df_year)

    return pd.concat(all_years, ignore_index=True)

def plot_yearly_boxplots(df_clustered):
    import matplotlib.pyplot as plt
    import seaborn as sns

    g = sns.catplot(
        data=df_clustered, 
        x='Cluster_KMeans', 
        y='MortalityRate', 
        col='Year', 
        col_wrap=4,
        kind='box', 
        height=3.5, 
        aspect=1
    )
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle('Mortality Rate by Cluster (KMeans) for Each Year')
    plt.show()

### 5/14/25, EB: It seems like the clustering does a decent job of separating out the counties that have high mortality rates. Now I'm interested in looking
### at the SVI profiles for each cluster, and seeing if we can identify any patterns in the SVI data that correspond to high-risk counties.
### In a perfect world, the profiles will line up with the results in mortality_autoencoder_model.py, but we'll see if that happens.
### The following function will take in the clustered data, and return a dataframe with the average SVI values for each cluster, for each year.

def profile_clusters_by_year(clustered_df, svi_vars, cluster_col='Cluster_KMeans'):
    """
    Computes mean SVI values for each cluster within each year.

    Args:
        clustered_df: DataFrame with Year, cluster labels, and SVI variables
        svi_vars: List of unprefixed SVI variable names (e.g., 'Below Poverty')
        cluster_col: Column name holding the cluster label (default: 'Cluster_KMeans')

    Returns:
        Multi-indexed DataFrame with (Year, Cluster) × SVI profile
    """
    grouped = (
        clustered_df.groupby(['Year', cluster_col])[svi_vars]
        .mean()
        .round(2)
        .sort_index()
    )
    return grouped

def compute_cluster_mortality_summary(clustered_df, cluster_col='Cluster_KMeans'):
    """
    Computes average mortality per cluster per year and flags highest-mortality cluster(s).

    Returns:
        DataFrame with (Year, Cluster) and MortalityRate stats + is_high_mortality boolean.
    """
    summary = (
        clustered_df
        .groupby(['Year', cluster_col])['MortalityRate']
        .agg(['mean', 'median'])
        .rename(columns={'mean': 'MeanMortality', 'median': 'MedianMortality'})
        .reset_index()
    )

    # Flag cluster(s) with highest mean mortality per year
    summary['is_high_mortality'] = summary.groupby('Year')['MeanMortality']\
                                          .transform(lambda x: x == x.max())

    return summary




def main():
    # # Load and prepare data
    # data_df = construct_data_df()
    # year = 2020
    # df, X_scaled, year_cols = prepare_clustering_data(data_df, year=year, scale_data=True)
    # df = run_clustering_algorithms(df, X_scaled, n_clusters=4)

    # ### Testing the clustering algorithms
    # # df_clusters = []
    # # for i in range(3, 6):
    # #     df = run_clustering_algorithms(df, X_scaled, n_clusters=i)
    # #     df_clusters.append(df)

    # #     # Plot mortality distributions by cluster
    # #     plot_cluster_mortality_distributions(df)

    # ### Profile KMeans clusters
    # svi_vars = [v for v in DATA if v != 'Mortality']
    # year_cols = [f'{year} {v}' for v in svi_vars]

    # # Subset to SVI columns + cluster label
    # svi_df = df[['FIPS', 'Cluster_KMeans'] + year_cols].dropna().copy()

    # # Rename for cleaner display
    # rename_map = {f'{year} {v}': v for v in svi_vars}
    # svi_df = svi_df.rename(columns=rename_map)

    # # Profile
    # cluster_profiles = profile_kmeans_clusters(svi_df, svi_vars)
    # #print(cluster_profiles)
    # cluster_profiles.to_csv('County Classification\Clustering\Cluster_KMeans_Profiles_2020.csv', index=True)


    ### 5/13/25, EB: Multi-year clustering
    # data_df = construct_data_df()
    # svi_vars = [v for v in DATA if v != 'Mortality']
    # long_df = reshape_multi_year_data(data_df, svi_vars, years=range(2010, 2023))
    # clustered_df = cluster_pooled_data(long_df, svi_vars, method='kmeans', n_clusters=4)
    # profiles = profile_kmeans_clusters(clustered_df, svi_vars)
    # profiles.to_csv('County Classification\Clustering\KMeans_MultiYear_Cluster_Profiles.csv')
    
    ### 5/13/25, EB: Year by year clustering
    data_df = construct_data_df()
    svi_vars = [v for v in DATA if v != 'Mortality']
    clustered_by_year_df = cluster_and_profile_by_year(data_df, svi_vars, years=range(2010, 2023), n_clusters=4)
    #plot_yearly_boxplots(clustered_by_year_df)
    cluster_profiles_by_year = profile_clusters_by_year(clustered_by_year_df, svi_vars)
    # cluster_profiles_by_year.to_csv('County Classification\Clustering\Yearly_KMeans_Cluster_SVI_Profiles.csv')
    # print('Cluster profiles by year saved to CSV.')
    ### Attempting to include a tag signifying the highest mortality cluster
    # Reset index for merging
    svi_profiles_reset = cluster_profiles_by_year.reset_index()
    mortality_summary = compute_cluster_mortality_summary(clustered_by_year_df)

    # Merge on Year and Cluster
    profile_with_mortality = pd.merge(svi_profiles_reset, mortality_summary,
                                    left_on=['Year', 'Cluster_KMeans'],
                                    right_on=['Year', 'Cluster_KMeans'])

    # Save to file
    profile_with_mortality.to_csv('County Classification\Clustering\Yearly_KMeans_Cluster_SVI_Profiles_with_Mortality.csv', index=False)
    print('Cluster profiles with mortality summary saved to CSV.') 



if __name__ == "__main__":
    main()

