### 5/1/25, EB: I talked with VM this morning, and we walked through everything I have tried so far. He likes the distribution, and thinks using it is a good idea,
### but he said I'm trying to make things too complicated. Why are we taking the SVI to predict the mortality/relative risk, to then go back to looking at the SVI?
### Instead, why don't we focus on the SVI variables themselves, and see what we can learn from them directly?
### So, in this script, I am going to try using anomaly detection on the SVI variables, and see if any patterns emerge. The idea is that if we can find anomalies in the SVI variables,
### they will maybe be associated with the high risk counties we are seeing in the mortality data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

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


def detect_anomalies_by_year(data_df, svi_vars, years=range(2010, 2023), contamination=0.05):
    """Detect anomalies in SVI space by year using Isolation Forest.
    
    Args:
        data_df: Wide-format dataframe from construct_data_df()
        svi_vars: List of SVI variable names (not year-prefixed)
        years: List of years to include (default 2010–2022)
        contamination: Proportion of anomalies to detect (0.05 = 5%)
    
    Returns:
        long_df: Long-format dataframe with Year, FIPS, SVI values, MortalityRate, Anomaly, and AnomalyScore
    """
    long_df = []

    for year in years:
        # Build column list
        year_svi_cols = [f"{year} {v}" for v in svi_vars if f"{year} {v}" in data_df.columns]
        if not year_svi_cols:
            continue
        
        required_cols = ['FIPS', f"{year} Mortality Rates"] + year_svi_cols
        temp_df = data_df[required_cols].dropna().copy()
        temp_df = temp_df.rename(columns={f"{year} Mortality Rates": "MortalityRate"})
        temp_df['Year'] = year

        # Rename SVI columns to raw names
        rename_dict = {f"{year} {v}": v for v in svi_vars if f"{year} {v}" in temp_df.columns}
        temp_df = temp_df.rename(columns=rename_dict)

        # Scale and detect anomalies
        X = temp_df[svi_vars].values
        X_scaled = StandardScaler().fit_transform(X)

        clf = IForest(contamination=contamination, random_state=42)
        clf.fit(X_scaled)

        temp_df['Anomaly'] = clf.predict(X_scaled)  # 1 = anomaly, 0 = normal
        temp_df['AnomalyScore'] = clf.decision_function(X_scaled)

        long_df.append(temp_df)

    return pd.concat(long_df, ignore_index=True)

def summarize_top_anomalies_by_year(anomaly_df, percentile=95):
    """
    Summarize the number of counties in the top X% of AnomalyScore for each year.
    
    Args:
        anomaly_df (pd.DataFrame): Output from detect_anomalies_by_year.
        percentile (float): Percentile threshold (e.g., 95 for top 5%).
    
    Returns:
        pd.DataFrame: Yearly summary with count of top-percentile anomalous counties.
    """
    results = []

    for year, group in anomaly_df.groupby('Year'):
        threshold = np.percentile(group['AnomalyScore'], percentile)
        top_counties = group[group['AnomalyScore'] >= threshold]

        results.append({
            'Year': year,
            'TotalCounties': len(group),
            f'Top{100 - percentile:.1f}PercentCount': len(top_counties),
            f'Top{100 - percentile:.1f}PercentShare': len(top_counties) / len(group)
        })

    return pd.DataFrame(results)

def get_persistent_anomalies(anomaly_df, percentile=95, min_years=1):
    """
    Identify counties that appear in the top X% of anomaly score in at least `min_years` years.

    Args:
        anomaly_df (pd.DataFrame): Output from detect_anomalies_by_year.
        percentile (float): Percentile cutoff for "top anomaly score" (default = 95 for top 5%).
        min_years (int): Minimum number of years a county must appear in top X% to be included.

    Returns:
        pd.DataFrame: FIPS, count of years in top X%, and optional metadata (e.g. urban_rural_class if present).
    """
    top_fips_by_year = []

    for year, group in anomaly_df.groupby('Year'):
        threshold = np.percentile(group['AnomalyScore'], percentile)
        top_fips = group[group['AnomalyScore'] >= threshold][['FIPS']]
        top_fips['Year'] = year
        top_fips_by_year.append(top_fips)

    # Combine all years and count appearances
    top_all_years = pd.concat(top_fips_by_year, ignore_index=True)
    counts = top_all_years['FIPS'].value_counts().reset_index()
    counts.columns = ['FIPS', 'TopPercentileYearCount']

    # Filter to those that appear at least min_years
    persistent = counts[counts['TopPercentileYearCount'] >= min_years].copy()

    return persistent

def plot_top_anomaly_mortality_scatter(anomaly_df, top_n=10, jitter_width=0.05):
    # Prepare data
    plot_data = []

    for year, year_df in anomaly_df.groupby('Year'):
        year_anomalies = year_df[year_df['Anomaly'] == 1]
        top_anomalies = year_anomalies.sort_values('AnomalyScore', ascending=False).head(top_n)

        for _, row in top_anomalies.iterrows():
            plot_data.append({
                'Year': year,
                'MortalityRate': row['MortalityRate'],
                'Type': 'Anomalous County'
            })

        # Add national average
        national_avg = year_df['MortalityRate'].mean()
        plot_data.append({
            'Year': year,
            'MortalityRate': national_avg,
            'Type': 'National Average'
        })

    plot_df = pd.DataFrame(plot_data)

    # Set up the plot
    plt.figure(figsize=(12, 7))

    for label, group in plot_df.groupby('Type'):
        if label == 'Anomalous County':
            # Add jitter to spread points horizontally
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(group))
            x_vals = group['Year'] + jitter
            plt.scatter(x_vals, group['MortalityRate'], label=label, alpha=0.7, color='darkred', s=50)
        else:
            # Plot national average without jitter
            plt.plot(group['Year'], group['MortalityRate'], label=label, color='black', linewidth=2, marker='o')

    plt.title(f'Mortality Rates of Top {top_n} Anomalous Counties per Year\nwith National Average')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate (per 100,000)')
    plt.grid(True, axis='y')
    plt.xticks(sorted(anomaly_df['Year'].unique()))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_top_anomaly_mortality_scatter_with_labels(anomaly_df, top_n=10, jitter_width=0.01):
    # Prepare data
    plot_data = []
    label_data = []

    for year, year_df in anomaly_df.groupby('Year'):
        year_anomalies = year_df[year_df['Anomaly'] == 1]
        top_anomalies = year_anomalies.sort_values('AnomalyScore', ascending=False).head(top_n)

        # Add all top N anomalies
        for _, row in top_anomalies.iterrows():
            plot_data.append({
                'Year': year,
                'FIPS': row['FIPS'],
                'MortalityRate': row['MortalityRate'],
                'Type': 'Anomalous County'
            })

        # Add label for highest mortality anomaly
        if not top_anomalies.empty:
            highest = top_anomalies.loc[top_anomalies['MortalityRate'].idxmax()]
            label_data.append({
                'Year': year,
                'MortalityRate': highest['MortalityRate'],
                'Label': highest['FIPS']  # or use a mapping to county name
            })

        # Add national average
        national_avg = year_df['MortalityRate'].mean()
        plot_data.append({
            'Year': year,
            'FIPS': 'National Avg',
            'MortalityRate': national_avg,
            'Type': 'National Average'
        })

    plot_df = pd.DataFrame(plot_data)
    label_df = pd.DataFrame(label_data)

    # Set up the plot
    plt.figure(figsize=(12, 7))

    # Plot anomalous counties with jitter
    anomalies = plot_df[plot_df['Type'] == 'Anomalous County']
    jitter = np.random.uniform(-jitter_width, jitter_width, size=len(anomalies))
    x_vals = anomalies['Year'] + jitter
    plt.scatter(x_vals, anomalies['MortalityRate'], label='Anomalous County', alpha=0.7, color='darkred', s=50)

    # Plot national average
    nat_avg = plot_df[plot_df['Type'] == 'National Average']
    plt.plot(nat_avg['Year'], nat_avg['MortalityRate'], color='black', linewidth=2, marker='o', label='National Average')

    # Add FIPS labels to highest-mortality anomalies
    for _, row in label_df.iterrows():
        plt.text(row['Year'] + 0.3, row['MortalityRate'], row['Label'],
                 fontsize=9, color='black', ha='left', va='center')

    plt.title(f'Mortality Rates of Top {top_n} Anomalous Counties per Year\nWith National Average and Highest-Anomaly Labels')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate (per 100,000)')
    plt.grid(True, axis='y')
    plt.xticks(sorted(anomaly_df['Year'].unique()))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_anomalies_mortality_scatter_with_labels(anomaly_df, jitter_width=0.02):
    # Prepare data
    plot_data = []
    label_data = []

    for year, year_df in anomaly_df.groupby('Year'):
        year_anomalies = year_df[year_df['Anomaly'] == 1]

        # Add all anomalies for that year
        for _, row in year_anomalies.iterrows():
            plot_data.append({
                'Year': year,
                'FIPS': row['FIPS'],
                'MortalityRate': row['MortalityRate'],
                'Type': 'Anomalous County'
            })

        # Label highest mortality among anomalies
        if not year_anomalies.empty:
            highest = year_anomalies.loc[year_anomalies['MortalityRate'].idxmax()]
            label_data.append({
                'Year': year,
                'MortalityRate': highest['MortalityRate'],
                'Label': highest['FIPS']
            })

        # Add national average
        national_avg = year_df['MortalityRate'].mean()
        plot_data.append({
            'Year': year,
            'FIPS': 'National Avg',
            'MortalityRate': national_avg,
            'Type': 'National Average'
        })

    plot_df = pd.DataFrame(plot_data)
    label_df = pd.DataFrame(label_data)

    # Plot setup
    plt.figure(figsize=(12, 7))

    # Plot all anomalies with jitter
    anomalies = plot_df[plot_df['Type'] == 'Anomalous County']
    jitter = np.random.uniform(-jitter_width, jitter_width, size=len(anomalies))
    x_vals = anomalies['Year'] + jitter
    plt.scatter(x_vals, anomalies['MortalityRate'], label='Anomalous County', alpha=0.7, color='darkred', s=50)

    # Plot national average
    nat_avg = plot_df[plot_df['Type'] == 'National Average']
    plt.plot(nat_avg['Year'], nat_avg['MortalityRate'], color='black', linewidth=2, marker='o', label='National Average')

    # Label highest mortality anomaly each year
    for _, row in label_df.iterrows():
        plt.text(row['Year'] + 0.3, row['MortalityRate'], row['Label'],
                 fontsize=9, color='black', ha='left', va='center')

    plt.title(f'Mortality Rates of All Anomalous Counties per Year\nWith National Average and Yearly Highest-Anomaly Labels')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate (per 100,000)')
    plt.grid(True, axis='y')
    plt.xticks(sorted(anomaly_df['Year'].unique()))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_anomalies_vs_high_mortality_scatter(anomaly_df, jitter_width=0.05, top_n=10):
    plot_data = []
    label_data = []

    for year, year_df in anomaly_df.groupby('Year'):
        # --- Anomalous Counties ---
        year_anomalies = year_df[year_df['Anomaly'] == 1]
        for _, row in year_anomalies.iterrows():
            plot_data.append({
                'Year': year,
                'MortalityRate': row['MortalityRate'],
                'FIPS': row['FIPS'],
                'Type': 'Anomalous County'
            })

        # Label the highest mortality anomaly
        if not year_anomalies.empty:
            highest = year_anomalies.loc[year_anomalies['MortalityRate'].idxmax()]
            label_data.append({
                'Year': year,
                'MortalityRate': highest['MortalityRate'],
                'Label': highest['FIPS']
            })

        # --- Top Mortality Counties ---
        top_mortality = year_df.sort_values('MortalityRate', ascending=False).head(top_n)
        for _, row in top_mortality.iterrows():
            plot_data.append({
                'Year': year,
                'MortalityRate': row['MortalityRate'],
                'FIPS': row['FIPS'],
                'Type': 'High Mortality County'
            })

        # --- National Average ---
        nat_avg = year_df['MortalityRate'].mean()
        plot_data.append({
            'Year': year,
            'MortalityRate': nat_avg,
            'FIPS': 'National Avg',
            'Type': 'National Average'
        })

    plot_df = pd.DataFrame(plot_data)
    label_df = pd.DataFrame(label_data)

    # Plot setup
    plt.figure(figsize=(12, 7))

    # Jittered scatter points
    for label, color in [('Anomalous County', 'darkred'), ('High Mortality County', 'blue')]:
        group = plot_df[plot_df['Type'] == label]
        jitter = np.random.uniform(-jitter_width, jitter_width, size=len(group))
        x_vals = group['Year'] + jitter
        plt.scatter(x_vals, group['MortalityRate'], label=label, alpha=0.7, color=color, s=50)

    # National average line
    nat_avg = plot_df[plot_df['Type'] == 'National Average']
    plt.plot(nat_avg['Year'], nat_avg['MortalityRate'], color='black', linewidth=2, marker='o', label='National Average')

    # Add labels for highest mortality anomaly each year
    for _, row in label_df.iterrows():
        plt.text(row['Year'] + 0.3, row['MortalityRate'], row['Label'],
                 fontsize=9, color='black', ha='left', va='center')

    plt.title(f'Mortality Rates by Year\nAnomalous Counties (Red), High-Mortality Counties (Blue), National Avg (Black)')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate (per 100,000)')
    plt.grid(True, axis='y')
    plt.xticks(sorted(anomaly_df['Year'].unique()))
    plt.legend()
    plt.tight_layout()
    plt.show()






def generalized_detect_anomalies_by_year(data_df, svi_vars, model_class, model_kwargs=None, years=range(2010, 2023), contamination=0.05):
    """
    Detect anomalies in SVI space by year using a specified PyOD model.
    
    Args:
        data_df: Wide-format dataframe from construct_data_df()
        svi_vars: List of SVI variable names (not year-prefixed)
        model_class: A PyOD class (e.g., IForest, KNN, LOF, AutoEncoder)
        model_kwargs: Optional dictionary of model-specific keyword args
        years: List of years to include
        contamination: Proportion of anomalies to detect
    
    Returns:
        long_df: Long-format dataframe with Year, FIPS, SVI values, MortalityRate, Anomaly, and AnomalyScore
    """
    if model_kwargs is None:
        model_kwargs = {}

    long_df = []

    for year in years:
        year_svi_cols = [f"{year} {v}" for v in svi_vars if f"{year} {v}" in data_df.columns]
        if not year_svi_cols:
            continue

        required_cols = ['FIPS', f"{year} Mortality Rates"] + year_svi_cols
        temp_df = data_df[required_cols].dropna().copy()
        temp_df = temp_df.rename(columns={f"{year} Mortality Rates": "MortalityRate"})
        temp_df['Year'] = year

        rename_dict = {f"{year} {v}": v for v in svi_vars if f"{year} {v}" in temp_df.columns}
        temp_df = temp_df.rename(columns=rename_dict)

        X = temp_df[svi_vars].values
        X_scaled = StandardScaler().fit_transform(X)

        # Instantiate and fit model
        model = model_class(contamination=contamination, **model_kwargs)
        model.fit(X_scaled)

        temp_df['Anomaly'] = model.predict(X_scaled)
        temp_df['AnomalyScore'] = model.decision_function(X_scaled)

        long_df.append(temp_df)

    return pd.concat(long_df, ignore_index=True)


### 5/7/25, EB: The following functions will be used in the PCA testing I'm doing.

def project_svi_to_pca_space(anomaly_df, svi_vars, n_components=5):
    X = anomaly_df[svi_vars].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Add PCA columns back to the original dataframe
    for i in range(n_components):
        anomaly_df[f'PC{i+1}'] = X_pca[:, i]

    return anomaly_df, pca

def detect_anomalies_on_pca(anomaly_df, n_components=5, contamination=0.05):
    pcs = [f'PC{i+1}' for i in range(n_components)]
    X = anomaly_df[pcs].dropna().values

    clf = IForest(contamination=contamination, random_state=42)
    clf.fit(X)

    # Assign anomaly results
    anomaly_df['Anomaly_PCA'] = clf.predict(X)
    anomaly_df['AnomalyScore_PCA'] = clf.decision_function(X)

    return anomaly_df




##### 5/5/25, EB: Testing anomaly map plotting

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === CONFIG ===
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
PRED_DIR = 'County Classification/Clustering/Anomaly_Detection_Results'
OUT_DIR = 'County Classification/Clustering/Anomaly_Detection_Maps'
os.makedirs(OUT_DIR, exist_ok=True)

# === Load county shapefile ===
def load_shapefile():
    shape = gpd.read_file(SHAPE_PATH)
    shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)
    return shape

def construct_persistent_anomaly_map(shape, persistent_df, out_path):
    # Merge the persistent anomaly info into the shapefile
    shape = shape.copy()
    persistent_df['FIPS'] = persistent_df['FIPS'].str.zfill(5)
    shape = shape.merge(persistent_df, on='FIPS', how='left')

    # All non-anomalous counties get 0
    shape['TopPercentileYearCount'] = shape['TopPercentileYearCount'].fillna(0)

    fig, main_ax = plt.subplots(figsize=(10, 5))
    plt.title('Persistent Anomalies in SVI Space (2010–2022)', size=16, weight='bold')

    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])

    state_boundaries = shape.dissolve(by='STATEFP', as_index=False)
    state_boundaries.boundary.plot(ax=main_ax, edgecolor='black', linewidth=.5)
    state_boundaries[state_boundaries['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=.5)
    state_boundaries[state_boundaries['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=.5)

    cmap = plt.get_cmap('viridis', 13)  # max 13 years
    bounds = np.arange(0, 14, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    shapes = [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax),
        (shape[shape['STATEFP'] == '02'], alaska_ax),
        (shape[shape['STATEFP'] == '15'], hawaii_ax)
    ]

    for inset, ax in shapes:
        inset.plot(ax=ax,
                   column='TopPercentileYearCount',
                   cmap=cmap,
                   edgecolor='black',
                   linewidth=0.1,
                   norm=norm,
                   missing_kwds={
                       "color": "lightgrey",
                       "edgecolor": "black",
                       "hatch": "///",
                       "label": "Not anomalous"
                   })

    for ax in [main_ax, alaska_ax, hawaii_ax]:
        ax.axis('off')

    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(bounds)
    cbar.set_label('# of Years in Top 5% Anomaly Score', fontsize=10, weight='bold')

    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()

# def plot_mortality_time_series(anomaly_df, persistent_df):
#     # Flag persistent anomalous counties
#     anomaly_df = anomaly_df.copy()
#     persistent_fips = persistent_df['FIPS'].unique()
#     anomaly_df['Persistent'] = anomaly_df['FIPS'].isin(persistent_fips)

#     # Group by Year and compute average mortality for persistent vs all
#     grouped = anomaly_df.groupby(['Year', 'Persistent'])['MortalityRate'].mean().reset_index()

#     # Pivot for plotting
#     grouped['Group'] = grouped['Persistent'].map({True: 'Persistent Anomalies', False: 'National Average'})

#     # Plot
#     plt.figure(figsize=(10,6))
#     sns.lineplot(data=grouped, x='Year', y='MortalityRate', hue='Group', marker='o')

#     plt.title('Opioid Mortality Over Time: Persistent Anomalies vs National Average')
#     plt.ylabel('Mortality Rate (per 100,000)')
#     plt.xlabel('Year')
#     plt.legend(title='Group')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def plot_mortality_time_series(anomaly_df, persistent_df):
    """ This function plots the mortality rates of persistently anomalous counties over time,
    along with the national average mortality rate for comparison. Rather than averaging the anomalous
    counties, like the previous function, this one plots each county's mortality rate as its own line.
    """

    anomaly_df = anomaly_df.copy()
    persistent_fips = persistent_df['FIPS'].unique()
    anomaly_df['Persistent'] = anomaly_df['FIPS'].isin(persistent_fips)

    # Filter to persistent anomaly counties
    persistent_df_full = anomaly_df[anomaly_df['Persistent']].copy()

    # Compute national average mortality by year
    national_avg = (
        anomaly_df
        .groupby('Year')['MortalityRate']
        .mean()
        .reset_index()
        .rename(columns={'MortalityRate': 'NationalAvg'})
    )

    # Merge the national average back to persistent counties for plotting
    plot_df = pd.merge(persistent_df_full, national_avg, on='Year', how='left')

    # Plot
    plt.figure(figsize=(12, 7))

    # Plot each anomalous county as a line
    for fips, county_df in plot_df.groupby('FIPS'):
        plt.plot(county_df['Year'], county_df['MortalityRate'], color='grey', alpha=0.4, linewidth=1)

    # Plot the national average as a bold line
    plt.plot(national_avg['Year'], national_avg['NationalAvg'], color='black', linewidth=3, label='National Average')

    plt.title('Opioid Mortality Trajectories\nPersistently Anomalous Counties vs National Average')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate (per 100,000)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mortality_time_series_highlighted(anomaly_df, persistent_df, top_n=5):
    import matplotlib.pyplot as plt
    import seaborn as sns

    anomaly_df = anomaly_df.copy()
    persistent_df = persistent_df.copy()

    # Identify top N most anomalous counties by how many years they appear in top 5%
    top_fips = persistent_df.sort_values('TopPercentileYearCount', ascending=False).head(top_n)['FIPS'].tolist()
    persistent_fips = persistent_df['FIPS'].unique()
    anomaly_df['Persistent'] = anomaly_df['FIPS'].isin(persistent_fips)
    anomaly_df['Highlight'] = anomaly_df['FIPS'].isin(top_fips)

    # Compute national average
    national_avg = (
        anomaly_df
        .groupby('Year')['MortalityRate']
        .mean()
        .reset_index()
        .rename(columns={'MortalityRate': 'NationalAvg'})
    )

    # Filter to persistent counties and merge in national average
    plot_df = anomaly_df[anomaly_df['Persistent']].copy()
    plot_df = pd.merge(plot_df, national_avg, on='Year', how='left')

    # Plot setup
    plt.figure(figsize=(12, 7))

    # Plot all persistent counties in grey
    for fips, county_df in plot_df[~plot_df['Highlight']].groupby('FIPS'):
        plt.plot(county_df['Year'], county_df['MortalityRate'], color='lightgrey', alpha=0.5, linewidth=1)

    # Highlight top N counties
    palette = sns.color_palette('tab10', top_n)
    for i, (fips, county_df) in enumerate(plot_df[plot_df['Highlight']].groupby('FIPS')):
        plt.plot(county_df['Year'], county_df['MortalityRate'], label=f'Top #{i+1}: {fips}', color=palette[i], linewidth=2)

    # National average in bold black
    plt.plot(national_avg['Year'], national_avg['NationalAvg'], color='black', linewidth=3, label='National Average')

    plt.title(f'Opioid Mortality: Top {top_n} Persistently Anomalous Counties vs National Average')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate (per 100,000)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mortality_time_series_highmortality_highanomaly(anomaly_df, persistent_df, top_n_anomalous=5, top_n_mortality=5):
    
    anomaly_df = anomaly_df.copy()
    persistent_df = persistent_df.copy()

    # Identify persistent and most anomalous counties
    persistent_fips = persistent_df['FIPS'].unique()
    top_anomalous_fips = (
        persistent_df.sort_values('TopPercentileYearCount', ascending=False)
        .head(top_n_anomalous)['FIPS'].tolist()
    )

    anomaly_df['Persistent'] = anomaly_df['FIPS'].isin(persistent_fips)
    anomaly_df['HighlightAnomalous'] = anomaly_df['FIPS'].isin(top_anomalous_fips)

    # Identify top N highest-mortality counties (cumulative over all years)
    avg_mortality = (
        anomaly_df.groupby('FIPS')['MortalityRate']
        .mean()
        .sort_values(ascending=False)
        .head(top_n_mortality)
        .index.tolist()
    )
    anomaly_df['HighlightHotspot'] = anomaly_df['FIPS'].isin(avg_mortality)
    print(f"Top {top_n_mortality} highest mortality counties: {avg_mortality}")
    
    # Compute national average
    national_avg = (
        anomaly_df.groupby('Year')['MortalityRate']
        .mean()
        .reset_index()
        .rename(columns={'MortalityRate': 'NationalAvg'})
    )

    # Filter to persistent anomaly counties
    #plot_df = anomaly_df[anomaly_df['Persistent']].copy()
    plot_df = anomaly_df.copy()
    plot_df = pd.merge(plot_df, national_avg, on='Year', how='left')

    # Plot setup
    plt.figure(figsize=(12, 7))

    # Plot all persistent counties (grey if not highlighted)
    for fips, county_df in plot_df[
        (~plot_df['HighlightAnomalous']) & (~plot_df['HighlightHotspot'])
    ].groupby('FIPS'):
        plt.plot(county_df['Year'], county_df['MortalityRate'],
                 color='lightgrey', alpha=0.5, linewidth=1)

    # Plot most anomalous counties in red
    for fips, county_df in plot_df[plot_df['HighlightAnomalous']].groupby('FIPS'):
        plt.plot(county_df['Year'], county_df['MortalityRate'],
                 color='red', linewidth=2, label=f'Anomalous: {fips}')

    # Plot highest mortality counties in blue
    for fips, county_df in plot_df[plot_df['HighlightHotspot']].groupby('FIPS'):
        plt.plot(county_df['Year'], county_df['MortalityRate'],
                 color='blue', linewidth=2, linestyle='--', label=f'High Mortality: {fips}')

    # National average in bold black
    plt.plot(national_avg['Year'], national_avg['NationalAvg'],
             color='black', linewidth=3, label='National Average')

    plt.title(f'Mortality Trends: Anomalies (red), Hot Spots (blue), National Avg (black)')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate (per 100,000)')
    plt.grid(True)

    # Avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            new_handles.append(h)
            new_labels.append(l)
    plt.legend(new_handles, new_labels)

    plt.tight_layout()
    plt.show()



#### 5/7/25, EB: Persistently High Mortality Anomaly Detection Functions

def plot_svi_mean_bar_comparison(hotspot_df, svi_vars):
    # Compute groupwise mean
    mean_svi = hotspot_df.groupby('Anomaly')[svi_vars].mean().T
    mean_svi.columns = ['Non-Anomalous', 'Anomalous']
    mean_svi['Difference'] = mean_svi['Anomalous'] - mean_svi['Non-Anomalous']
    mean_svi = mean_svi.sort_values(by='Difference', ascending=False)

    # Plot
    mean_svi[['Non-Anomalous', 'Anomalous']].plot(kind='bar', figsize=(12, 6))
    plt.title('Mean SVI Values: Anomalous vs Non-Anomalous\n(Persistent High-Mortality Counties)')
    plt.ylabel('Mean (Standardized)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    return mean_svi  # optionally return this for inspection

def plot_hotspot_mortality_time_series(hotspot_df, full_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    hotspot_df = hotspot_df.copy()
    full_df = full_df.copy()

    # Merge national average for context
    national_avg = (
        full_df.groupby('Year')['MortalityRate']
        .mean()
        .reset_index()
        .rename(columns={'MortalityRate': 'NationalAvg'})
    )

    plot_df = pd.merge(hotspot_df, national_avg, on='Year', how='left')

    # Plot setup
    plt.figure(figsize=(12, 7))

    # Plot non-anomalous hotspots in grey
    for fips, county_df in plot_df[plot_df['Anomaly'] == 0].groupby('FIPS'):
        plt.plot(county_df['Year'], county_df['MortalityRate'], color='lightgrey', alpha=0.5, linewidth=1)

    # Plot anomalous hotspots in red
    for fips, county_df in plot_df[plot_df['Anomaly'] == 1].groupby('FIPS'):
        plt.plot(county_df['Year'], county_df['MortalityRate'], color='red', linewidth=2, label=f'Anomalous: {fips}')

    # Plot national average
    plt.plot(national_avg['Year'], national_avg['NationalAvg'], color='black', linewidth=3, label='National Average')

    plt.title('Mortality Time Series: Anomalous vs Non-Anomalous High-Mortality Counties')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate (per 100,000)')
    plt.grid(True)

    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            new_handles.append(h)
            new_labels.append(l)
    plt.legend(new_handles, new_labels)

    plt.tight_layout()
    plt.show()


def main():
    
    ### 5/1/25, EB: Here I am beginning the investigation into the anomalous counties.
    ### anomaly_df contains the anomalies detected in the SVI variables, and persistent_df contains the counties that are persistently anomalous, i.e.
    ### the ones that are in the top 5% of anomaly scores for at least 10 years.
    data_df = construct_data_df()
    svi_vars = [v for v in DATA if v != 'Mortality']

    anomaly_df = detect_anomalies_by_year(data_df, svi_vars)
    anomaly_df = generalized_detect_anomalies_by_year(data_df, svi_vars, model_class=KNN, years=range(2010, 2023), contamination=0.05)
    #plot_all_anomalies_mortality_scatter_with_labels(anomaly_df)
    plot_anomalies_vs_high_mortality_scatter(anomaly_df, top_n=25)
    
    
    ### 5/8/25, EB: Consistently finding that anomalous counties are not the same as high-mortality counties. I'm going to look at counties with low anomaly scores, to see if they are high-mortality.
    lowest_score_df = (
        anomaly_df.sort_values('AnomalyScore')
        .groupby('Year')
        .head(20)
    )

    lowest_score_summary = (
        lowest_score_df.groupby('Year')['MortalityRate']
        .describe()
    )
    print('20 Lowest Anomaly Score Counties by Year, Statistics:')
    print(lowest_score_summary)
 
    
    #anomaly_df = generalized_detect_anomalies_by_year(data_df, svi_vars, model_class=IForest, years=range(2010, 2023), contamination=0.05)
    #print(anomaly_df.head())
    #anomaly_summary = summarize_top_anomalies_by_year(anomaly_df, percentile=95)
    #print(anomaly_summary)
    
    #persistent_df = get_persistent_anomalies(anomaly_df, percentile=95, min_years=10)

    # # pd.save_csv(persistent_df, 'County Classification/Clustering/Anomaly_Detection_Results/persistent_anomalies.csv', index=False)
    # #print(persistent_df.sort_values('TopPercentileYearCount', ascending=False))
    # enriched = pd.merge(persistent_df, data_df[['FIPS', 'urban_rural_class']].drop_duplicates(), on='FIPS', how='left')
    # print(enriched.head())
    # print("Persistently Anomalous Counties:")
    # print(enriched.sort_values('TopPercentileYearCount', ascending=False))
    # #print("Unique classes:", enriched['urban_rural_class'].unique())
    
    # counts = enriched['urban_rural_class'].value_counts().sort_index()
    # print(counts)

#####################################################################
    ### 5/5/25, EB: Here I'm looking at the mortality rates of the persistently anomalous counties, stratified by urbanicity class.
    # anomaly_df['Persistent'] = anomaly_df['FIPS'].isin(persistent_df['FIPS'])

    # # Example: Average mortality by urbanicity class within persistent anomalies
    # anomalous_mortality = anomaly_df[anomaly_df['Persistent']].groupby('urban_rural_class')['MortalityRate'].mean()
    # print(anomalous_mortality)
    # Ensure FIPS is consistent format

    # anomaly_df['FIPS'] = anomaly_df['FIPS'].str.zfill(5)
    # data_df['FIPS'] = data_df['FIPS'].str.zfill(5)

    # # Select unique FIPS-to-urbanicity mapping
    # urbanicity_lookup = data_df[['FIPS', 'urban_rural_class']].drop_duplicates()

    # # Merge into anomaly_df
    # anomaly_df = pd.merge(anomaly_df, urbanicity_lookup, on='FIPS', how='left')
    # persistent_fips = persistent_df['FIPS'].unique()
    # anomaly_df['Persistent'] = anomaly_df['FIPS'].isin(persistent_fips)

    # urban_stratified = (
    #     anomaly_df[anomaly_df['Persistent']]
    #     .groupby('urban_rural_class')['MortalityRate']
    #     .mean()
    #     .sort_index()
    # )

    # print(urban_stratified)




########################################################
### Here I compare the average SVI values of the persistent anomalies to the rest of the counties.

    # # Merge persistent flags back into the full anomaly_df
    # persistent_fips = persistent_df['FIPS'].unique()
    # anomaly_df['Persistent'] = anomaly_df['FIPS'].isin(persistent_fips)

    # # Compute groupwise means over all years
    # feature_means = anomaly_df.groupby('Persistent')[svi_vars].mean().T
    # feature_means.columns = ['Non-Anomalous', 'Persistent Anomalous']
    # feature_means['Difference'] = feature_means['Persistent Anomalous'] - feature_means['Non-Anomalous']
    # feature_means = feature_means.sort_values(by='Difference', ascending=False)

    # import matplotlib.pyplot as plt

    # feature_means[['Non-Anomalous', 'Persistent Anomalous']].plot(kind='bar', figsize=(10,6))
    # plt.title('SVI Feature Comparison: Persistent Anomalies vs Others')
    # plt.ylabel('Mean (Standardized)')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()
########################################################
### Here I perform a t-test to see if the persistent anomalies are significantly different from the rest of the counties.
    # from scipy.stats import ttest_ind

    # stats = []
    # for var in svi_vars:
    #     group1 = anomaly_df[anomaly_df['Persistent']][var]
    #     group0 = anomaly_df[~anomaly_df['Persistent']][var]
    #     stat, pval = ttest_ind(group1, group0, equal_var=False)
    #     stats.append({'Variable': var, 'T-stat': stat, 'p-value': pval})

    # stats_df = pd.DataFrame(stats).sort_values('p-value')
    # print(stats_df)


##################################################################
# ### 5/5/25, EB: Here I am trying to get a visualization of the persistently anomalous counties.
# ### I've adapted the script EB_rel_risk_regression_maps.py to make this map plotting script.
#     shape = load_shapefile()
#     data_df = construct_data_df()
#     svi_vars = [v for v in DATA if v != 'Mortality']
#     anomaly_df = detect_anomalies_by_year(data_df, svi_vars)
#     persistent_df = get_persistent_anomalies(anomaly_df, percentile=95, min_years=1)

#     out_path = f'{OUT_DIR}/persistent_anomalies_map.png'
#     construct_persistent_anomaly_map(shape, persistent_df, out_path)
#     print(f"✅ Map saved to {out_path}")


# ### Here I am plotting the mortality rates of the persistently anomalous counties, along with the national average each year.

#     plot_mortality_time_series_highlighted(anomaly_df, persistent_df, top_n=10)
#     plot_mortality_time_series_highmortality_highanomaly(anomaly_df, persistent_df, top_n_anomalous=10, top_n_mortality=10)




######################################################################
### 5/7/25, EB: So the above analysis revealed that the persistently anomalous counties are anomalous in Minority Status, Crowding, No Vehicle (above average),
### and Mobile Homes, Single-Parent Household, and Aged 65 or Older (below average). The t-test results also show that these variables are significantly different from the rest of the counties.
### This however did not reveal any serious patterns in the mortality rates of these counties; the most anomalous all had below average mortality rates. Some were pretty high, but they weren't the most anomalous.
### VM suggested I'm essentially looking at the inverse of what I should be, looking at (1 - \alpha) instead of \alpha itself. He also suggested I try doing PCA before anomaly detection, to see if that reveals anything interesting.
### The following will be exploring some new approaches to this.

    # data_df = construct_data_df()
    # svi_vars = [v for v in DATA if v != 'Mortality']

    # # anomaly_df = detect_anomalies_by_year(data_df, svi_vars)
    # anomaly_df = generalized_detect_anomalies_by_year(data_df, svi_vars, model_class=KNN, model_kwargs={'n_neighbors': 10} ,years=range(2010, 2023), contamination=0.05)

    # # Step 1: Project to PCA space
    # anomaly_df_pca, pca_model = project_svi_to_pca_space(anomaly_df, svi_vars, n_components=4)

    # # Step 2: Detect anomalies in PCA space
    # anomaly_df_pca = detect_anomalies_on_pca(anomaly_df_pca, n_components=4, contamination=0.05)
    
    # # Compare average mortality for new anomalies
    # mortality_compare = (
    #     anomaly_df_pca
    #     .groupby('Anomaly_PCA')['MortalityRate']
    #     .mean()
    # )
    # print(mortality_compare)
    
    # # sns.boxplot(data=anomaly_df_pca, x='Anomaly_PCA', y='MortalityRate')
    # # plt.xticks([0, 1], ['Not Anomalous', 'Anomalous (PCA)'])
    # # plt.title("Mortality Rates in PCA-Based Anomaly Groups")
    # # plt.show()


    
    #persistent_df = get_persistent_anomalies(anomaly_df, percentile=95, min_years=10)



##########################################################################################################
### 5/7/25, EB: Ok, so it seems like no matter what I try with the general anomaly detection, the anomalous counties end
### up being the cold spots, the counties with below average mortality rates.
### I thought about it and I want to try going back to when I looked at the persistently high-risk counties, and seeing what sorts of anomalies they might have.
### Maybe I can tease out some sort of high-risk profile in the SVI variables, and can use those as predictors for the mortality rates.

#     data_df = construct_data_df()
#     svi_vars = [v for v in DATA if v != 'Mortality']

#     anomaly_df = detect_anomalies_by_year(data_df, svi_vars)
#     #persistent_df = get_persistent_anomalies(anomaly_df, percentile=95, min_years=10)

#     # For example: top 10% in at least 10 out of 13 years
#     thresholds = (
#         anomaly_df.groupby('Year')['MortalityRate']
#         .quantile(0.90)
#         .to_dict()
#     )

#     anomaly_df['HighMortality'] = anomaly_df.apply(
#         lambda row: row['MortalityRate'] >= thresholds[row['Year']],
#         axis=1
#     )

#     # Count years each county was in the top 10%
#     high_mortality_counts = (
#         anomaly_df[anomaly_df['HighMortality']]
#         .groupby('FIPS')
#         .size()
#         .reset_index(name='HighMortalityYears')
#     )

#     # Persistent high-mortality counties
#     persistent_hot_fips = high_mortality_counts[high_mortality_counts['HighMortalityYears'] >= 10]['FIPS'].tolist()

#     # Filter the full dataset to only those counties
#     hotspot_df = anomaly_df[anomaly_df['FIPS'].isin(persistent_hot_fips)].copy()

#     # Run PCA or not, your choice
#     from pyod.models.iforest import IForest
#     #from sklearn.preprocessing import StandardScaler

#     X = hotspot_df[svi_vars].dropna()
#     #X_scaled = StandardScaler().fit_transform(X)

#     model = IForest(contamination=0.1)
#     model.fit(X)

#     hotspot_df['Anomaly'] = model.predict(X)
#     hotspot_df['AnomalyScore'] = model.decision_function(X)

# #    svi_bar_df = plot_svi_mean_bar_comparison(hotspot_df, svi_vars)
#     plot_hotspot_mortality_time_series(hotspot_df, anomaly_df)


#     # print(hotspot_df.head())
#     # # Save the results
#     # hotspot_df.to_csv('County Classification/Clustering/Anomaly_Detection_Results/persistently_hotspot_anomaly_analysis.csv', index=False)
#     # print("✅ Hotspot anomalies saved to persistent_hotspot_anomalies.csv")

if __name__ == "__main__":
    main()



