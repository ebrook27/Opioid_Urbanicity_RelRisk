### 4/11/25, EB: Dr. Maroulas wants me to investigate the high risk counties directly. I think that we might be able to take some of these results and incorporate them into the classification model
### eventually, but I want to investigate a few things first.
### I found in Data\Mortality\top_mortality_counties.py that the counties with the top 5% of mortality stays roughly constant over the course of the study. We find that 101 counties appear
### in the top 5% for at least 10 of the 13 years we have data for. My goal with this script is to take a look at these counties, and see if we can find any patterns in the data.
### Here I am going to try to cluster the SVI data for these counties, and then apply the urbanicity label to them, to see what we find.

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import seaborn as sns
# import matplotlib.pyplot as plt


# # === Load the persistent FIPS list ===
# persistent_fips = pd.read_csv("Data\Mortality\Final Files\Mortality_top10_percent_counties_10yrs.csv", dtype={'FIPS': str})
# persistent_fips['FIPS'] = persistent_fips['FIPS'].str.zfill(5)
# persistent_fips_set = set(persistent_fips['FIPS'])


# # === Load SVI variables ===
# svi_vars = ['Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
#             'Group Quarters', 'Limited English Ability', 'Minority Status',
#             'Mobile Homes', 'Multi-Unit Structures', 'No High School Diploma',
#             'No Vehicle', 'Single-Parent Household', 'Unemployment']

# svi_dfs = []

# for var in svi_vars:
#     var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
#     var_df = pd.read_csv(var_path, dtype={'FIPS': str})
#     var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
#     # Take most recent year only (2022)
#     var_df = var_df[['FIPS', '2022 ' + var]].rename(columns={f'2022 {var}': var})
#     svi_dfs.append(var_df)

# # Merge all SVI variables
# from functools import reduce
# svi_df = reduce(lambda left, right: pd.merge(left, right, on='FIPS', how='inner'), svi_dfs)

# # Filter to just the persistent counties
# svi_df = svi_df[svi_df['FIPS'].isin(persistent_fips_set)].set_index('FIPS')

# #print(svi_df.head()) 

# # Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(svi_df)

# # Cluster (you can experiment with different n_clusters)
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(X_scaled)
# svi_df['Cluster'] = clusters

# # === Load county class labels ===
# nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
# nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
# nchs_df = nchs_df[['FIPS', '2023 Code']].rename(columns={'2023 Code': 'Urban_Class'})

# # Merge with cluster results
# svi_df = svi_df.merge(nchs_df, left_index=True, right_on='FIPS', how='left')

# # === Analyze clusters vs urban class ===
# cluster_summary = svi_df.groupby(['Cluster', 'Urban_Class']).size().unstack(fill_value=0)

# print("📊 Cluster composition by Urban Class:")
# print(cluster_summary)

# # === Optional: visualize SVI cluster profiles ===
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=svi_df, x='Cluster', y='Below Poverty')
# plt.title("Distribution of 'Below Poverty' by Cluster")
# plt.show()

######################################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################################
### 4/15/25, EB: The following tries to use the elbow and silhouette methods to determine the optimal number of clusters.
### Pretty inconclusive, but 3 didn't seem bad to use, so I'm sticking with it.

# inertias = []
# silhouettes = []
# k_range = range(2, 10)

# for k in k_range:
#     km = KMeans(n_clusters=k, random_state=42)
#     labels = km.fit_predict(X_scaled)
    
#     inertias.append(km.inertia_)
#     silhouettes.append(silhouette_score(X_scaled, labels))

# # Plotting Elbow Method
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(k_range, inertias, marker='o')
# plt.title("Elbow Method (Inertia)")
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia")

# # Plotting Silhouette Score
# plt.subplot(1, 2, 2)
# plt.plot(k_range, silhouettes, marker='s')
# plt.title("Silhouette Score")
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Score")
# plt.tight_layout()
# plt.show()
######################################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################################

### 4/16/25, EB: Here I am trying to expand the original script above to cluster the data for the persistent counties for each year, and not just the latest year in the study.
### This way we will get a better feel for the whole data, not just one year.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import os
os.environ["OMP_NUM_THREADS"] = "1"


# === Config ===
YEARS = list(range(2010, 2023))
N_CLUSTERS = 3

# === Load persistent top-10% FIPS list ===
persistent_fips = pd.read_csv("Data/Mortality/Final Files/Mortality_top10_percent_counties_10yrs_lognormal.csv", dtype={'FIPS': str})
persistent_fips['FIPS'] = persistent_fips['FIPS'].str.zfill(5)
persistent_fips_set = set(persistent_fips['FIPS'])

# === SVI Variables ===
svi_vars = ['Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
            'Group Quarters', 'Limited English Ability', 'Minority Status',
            'Mobile Homes', 'Multi-Unit Structures', 'No High School Diploma',
            'No Vehicle', 'Single-Parent Household', 'Unemployment']

# === Load county class labels ===
nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
nchs_df = nchs_df[['FIPS', '2023 Code']].rename(columns={'2023 Code': 'Urban_Class'})

# === Loop over years ===
all_clusters = []
cluster_summaries = {}

for year in YEARS:
    print(f"\n📅 Clustering for year {year}...")

    svi_dfs = []
    for var in svi_vars:
        path = f'Data/SVI/Final Files/{var}_final_rates.csv'
        df = pd.read_csv(path, dtype={'FIPS': str})
        df['FIPS'] = df['FIPS'].str.zfill(5)
        if f'{year} {var}' not in df.columns:
            print(f"⚠️ Missing {year} data for {var}, skipping.")
            continue
        df = df[['FIPS', f'{year} {var}']].rename(columns={f'{year} {var}': var})
        svi_dfs.append(df)

    # Merge all SVI variables for this year
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on='FIPS', how='inner'), svi_dfs)

    # Filter to persistent counties
    svi_merged = svi_merged[svi_merged['FIPS'].isin(persistent_fips_set)].set_index('FIPS')

    if svi_merged.empty:
        print("❌ No data to cluster.")
        continue

    # Standardize and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(svi_merged)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    svi_merged['Cluster'] = clusters
    svi_merged['Year'] = year

    # Merge urbanicity
    svi_merged = svi_merged.merge(nchs_df, left_index=True, right_on='FIPS', how='left')
    
        # Summarize cluster composition by urban class for the year
    summary = (
        svi_merged.groupby(['Cluster', 'Urban_Class'])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(f"\n📊 Cluster composition for year {year}:")
    print(summary)
    cluster_summaries[year] = summary

    all_clusters.append(svi_merged[['FIPS', 'Year', 'Cluster', 'Urban_Class']])

# === Combine all years ===
cluster_df = pd.concat(all_clusters).reset_index(drop=True)

# Combine all summaries into one long dataframe
combined_summary = pd.concat(cluster_summaries, names=['Year', 'Cluster']).reset_index()
combined_summary.to_csv("County Classification/Cluster_Urban_Class_Summary.csv", index=False)

# === Save FIPS + Cluster assignments for analysis ===
cluster_df.to_csv("County Classification/Persistent_Counties_Cluster_Results.csv", index=False)
print("✅ Cluster results saved to: County Classification/Persistent_Counties_Cluster_Results.csv")


# # === Optional: visualize cluster stability across years ===
# pivot = cluster_df.pivot(index='FIPS', columns='Year', values='Cluster')
# print("\n📊 Cluster assignment over time:")
# print(pivot.head())

# # === Plot cluster trajectories (optional) ===
# plt.figure(figsize=(12, 6))
# sns.heatmap(pivot, cmap='Set2', cbar=False)
# plt.title("🌀 Cluster Assignments of Persistent Counties (2010–2022)")
# plt.xlabel("Year")
# plt.ylabel("County (FIPS)")
# plt.tight_layout()
# plt.show()
