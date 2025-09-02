### 5/14/25, EB: I have gotten results from both the clustering and autoencoder models. My goal with
### this script is to compare the SVI profiles of the two models. Specifically, I want to compare the SVI
### profiles of the highest-mortality cluster each year with the SVI profiles from the hotspots identified
### by the autoencoder. In an ideal scenario, there will be a lot of overlap between the two.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load both datasets ===
auto_df = pd.read_csv('County Classification/autoencoder_svi_high_error_comparison_by_year.csv')
cluster_df = pd.read_csv('County Classification/Clustering/Yearly_KMeans_Cluster_SVI_Profiles_with_Mortality.csv')

# === Filter to high-mortality clusters only ===
high_mortality_clusters = cluster_df[cluster_df['is_high_mortality'] == True].copy()

# === Identify SVI variable columns ===
exclude_cols = ['Year', 'Cluster_KMeans', 'MeanMortality', 'MedianMortality', 'is_high_mortality']
svi_vars = [col for col in high_mortality_clusters.columns if col not in exclude_cols]

# === Reshape the cluster data to long format ===
cluster_long = high_mortality_clusters.melt(
    id_vars=['Year'], value_vars=svi_vars,
    var_name='Variable', value_name='ClusterMean'
)

# === Clean up autoencoder variable names ===
auto_df = auto_df.rename(columns={
    'Feature': 'Variable',
    'High Error Mean': 'HighErrorMean'
})
auto_df['Variable'] = auto_df['Variable'].str.replace(r'^\d{4} ', '', regex=True)

# === Merge on Year and Variable ===
auto_subset = auto_df[['Year', 'Variable', 'HighErrorMean']].copy()
comparison_df = pd.merge(cluster_long, auto_subset, on=['Year', 'Variable'], how='inner')

# === Compute the difference ===
comparison_df['Difference'] = comparison_df['ClusterMean'] - comparison_df['HighErrorMean']

# # === Save the result ===
# comparison_df.to_csv("County Classification/Clustering/Cluster_vs_Autoencoder_SVI_Comparison.csv", index=False)

# print("✅ Comparison file saved: Cluster_vs_Autoencoder_SVI_Comparison.csv")


# === Compute average absolute difference per variable ===
summary = (
    comparison_df.groupby("Variable")["Difference"]
    .apply(lambda x: x.abs().mean())
    .reset_index()
    .rename(columns={"Difference": "AvgAbsDifference"})
    .sort_values("AvgAbsDifference", ascending=False)
)

# === Plot bar chart ===
plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x="AvgAbsDifference", y="Variable", palette="viridis")

plt.title("Average Absolute Difference Between Cluster and Autoencoder Profiles")
plt.xlabel("Mean Absolute Difference (0–100 scale)")
plt.ylabel("SVI Variable")
plt.grid(axis='x')
plt.tight_layout()
plt.show()

####################################################################################################################################################################################################
### 5/15/25, EB: The following code is similar to the above, but I break the comparison down by urbanicity category as well.
### 5/15/25, 3:10pm, EB: Need to alter the other algorithms to include FIPS to merge with urbanicity data.

import pandas as pd

# === Load urbanicity labels ===
urban_df = pd.read_csv("NCHS_urban_v_rural.csv", dtype={'FIPS': str})
urban_df['FIPS'] = urban_df['FIPS'].str.zfill(5)
urban_df = urban_df.rename(columns={'2023 Code': 'UrbanicityClass'})

# Optional: map to readable labels
urbanicity_map = {
    1: 'Large Central Metro',
    2: 'Large Fringe Metro',
    3: 'Medium Metro',
    4: 'Small Metro',
    5: 'Micropolitan',
    6: 'Non-Core'
}
urban_df['UrbanicityLabel'] = urban_df['UrbanicityClass'].map(urbanicity_map)

# === Load clustering results ===
cluster_df = pd.read_csv("Yearly_KMeans_Cluster_SVI_Profiles_with_Mortality.csv")
cluster_df['FIPS'] = cluster_df['FIPS'].astype(str).str.zfill(5)

# Merge urbanicity into cluster data
cluster_df = pd.merge(cluster_df, urban_df, on='FIPS', how='left')

# === Load autoencoder high-error results ===
ae_df = pd.read_csv("autoencoder_svi_high_error_comparison_by_year.csv")
ae_df['FIPS'] = ae_df['FIPS'].astype(str).str.zfill(5)

# Merge urbanicity into AE data
ae_df = pd.merge(ae_df, urban_df, on='FIPS', how='left')

# === Group and compare by urbanicity ===
# Example: compute average SVI profiles by urbanicity and year
cluster_summary = (
    cluster_df
    .groupby(['Year', 'UrbanicityLabel'])
    .mean(numeric_only=True)
    .reset_index()
)

ae_summary = (
    ae_df[ae_df['High Error'] == True]  # filter to high-error counties
    .groupby(['Year', 'UrbanicityLabel'])
    .mean(numeric_only=True)
    .reset_index()
)

# === Export for further analysis or visualization ===
cluster_summary.to_csv("Cluster_SVI_By_Urbanicity.csv", index=False)
ae_summary.to_csv("AE_HighError_SVI_By_Urbanicity.csv", index=False)

print("✅ Urbanicity-stratified SVI summaries saved.")
