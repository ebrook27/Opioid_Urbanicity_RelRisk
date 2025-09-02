### 5/21/25, EB: Got a working RF model to predict mortality each year, using SVI + urbanicity. Here we produce maps of the accuracy of the model for each year.

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Reuse this from existing script
def prepare_yearly_prediction_data():
    DATA = ['Mortality', 'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
            'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
            'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
            'Single-Parent Household', 'Unemployment']

    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2023))

    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

    mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
    mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
    mort_df = mort_df.set_index('FIPS')

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

    from functools import reduce
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    for y in years:
        mort_col = f'{y+1} MR'
        if mort_col not in mort_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'mortality_rate'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(mort_df[mort_col])

    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')
    svi_merged = svi_merged.dropna()

    return svi_merged

# === ACCURACY MAP FUNCTIONS ===
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
PRED_DIR = 'County Classification/RF_Mortality_Testing'
OUT_DIR = 'County Classification/RF_Mortality_Accuracy_Maps'
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def run_rf_and_save_preds(df):
    for year in range(2010, 2023):
        print(f"📅 Running RF for year {year}...")
        df_year = df[df['year'] == year].copy()
        if df_year.empty:
            continue

        X = df_year.drop(columns=['FIPS', 'year', 'mortality_rate'])
        y = df_year['mortality_rate']

        model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
        model.fit(X, y)
        y_pred = model.predict(X)

        df_preds = df_year[['FIPS']].copy()
        df_preds[f'{year}_True_MR'] = y.values
        df_preds[f'{year}_Pred_MR'] = y_pred
        df_preds.to_csv(f"{PRED_DIR}/{year}_MR_predictions.csv", index=False)


def load_shapefile():
    shape = gpd.read_file(SHAPE_PATH)
    shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)
    return shape

def load_yearly_preds(year):
    file_path = f'{PRED_DIR}/{year}_MR_predictions.csv'
    df = pd.read_csv(file_path, dtype={'FIPS': str})
    df['FIPS'] = df['FIPS'].str.zfill(5)
    df['Absolute_Error'] = abs(df[f'{year}_True_MR'] - df[f'{year}_Pred_MR'])
    max_err = df['Absolute_Error'].max()
    df['Accuracy'] = 1 - (df['Absolute_Error'] / max_err)
    df['Accuracy'] = df['Accuracy'].clip(lower=0.0001, upper=0.9999)
    return df[['FIPS', 'Accuracy']]

def merge_data(shape, acc_df):
    return shape.merge(acc_df, on='FIPS', how='left')

def construct_accuracy_map(shape, year):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    plt.title(f'{year} RF Accuracy Map', size=16, weight='bold')

    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4]) 
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])  

    state_boundaries = shape.dissolve(by='STATEFP', as_index=False)
    state_boundaries.boundary.plot(ax=main_ax, edgecolor='black', linewidth=.5)
    state_boundaries[state_boundaries['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=.5)
    state_boundaries[state_boundaries['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=.5)

    cmap = plt.get_cmap('RdYlGn', 20)
    shapes = [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax),
        (shape[shape['STATEFP'] == '02'], alaska_ax),
        (shape[shape['STATEFP'] == '15'], hawaii_ax)
    ]

    for inset, ax in shapes:
        inset.plot(ax=ax, column='Accuracy', cmap=cmap, edgecolor='black', linewidth=0.1)

    for ax in [main_ax, alaska_ax, hawaii_ax]:
        ax.axis('off')

    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

    bounds = np.linspace(0, 1, 21)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(bounds)
    cbar.ax.set_yticklabels([f'{int(b*100)}%' for b in bounds])
    cbar.set_label('Accuracy', fontsize=10, weight='bold')

    plt.savefig(f'{OUT_DIR}/RF_Accuracy_Map_{year}.png', bbox_inches='tight', dpi=300)
    plt.close()

def run_accuracy_map_pipeline():
    shape = load_shapefile()
    for year in range(2010, 2023):
        print(f"🗺️ Generating map for {year}...")
        preds = load_yearly_preds(year)
        merged = merge_data(shape, preds)
        construct_accuracy_map(merged, year)
    print("✅ All maps saved.")

if __name__ == "__main__":
    df = prepare_yearly_prediction_data()
    run_rf_and_save_preds(df)
    run_accuracy_map_pipeline()
