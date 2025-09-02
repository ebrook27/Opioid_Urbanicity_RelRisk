### 5/21/25, EB: Ok, so we've got working models for both the whole country, and for each county category. We've got feature importance rankings
### for both, and we have accuracy maps for the whole country. Here I'm trying to make accuracy maps for each county category, each year.
### I want all of the non-category counties that year to be greyed out, and only have the accurayc maps for the counties in that category.

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
PRED_DIR = 'County Classification/RF_Mort_Preds_by_Urbanicity'
OUT_DIR = 'County Classification/RF_Mortality_Accuracy_Maps/Urbanicity_RF_Accuracy_Plots'
os.makedirs(OUT_DIR, exist_ok=True)

# === Load county shapefile ===
def load_shapefile():
    shape = gpd.read_file(SHAPE_PATH)
    shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)
    return shape

# === Load predictions for one year and category ===
def load_yearly_preds(year, county_class):
    file_path = f'{PRED_DIR}/{year}_Cat_{county_class}_MR_predictions.csv'
    df = pd.read_csv(file_path, dtype={'FIPS': str})
    df['FIPS'] = df['FIPS'].str.zfill(5)
    df['Absolute_Error'] = abs(df['True'] - df['Predicted'])
    max_err = df['Absolute_Error'].max()
    df['Accuracy'] = 1 - (df['Absolute_Error'] / max_err)
    df['Accuracy'] = df['Accuracy'].clip(lower=0.0001, upper=0.9999)
    return df[['FIPS', 'Accuracy']]

# === Load county class data ===
def load_county_classes():
    class_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    class_df['FIPS'] = class_df['FIPS'].str.zfill(5)
    class_df['county_class'] = class_df['2023 Code'].astype(str)
    return class_df[['FIPS', 'county_class']]

# === Merge with shapefile ===
def merge_data(shape, acc_df, county_class, class_df):
    shape = shape.merge(class_df, on='FIPS', how='left')
    shape = shape.merge(acc_df, on='FIPS', how='left')
    shape['PlotValue'] = shape.apply(lambda row: row['Accuracy'] if row['county_class'] == county_class else np.nan, axis=1)
    return shape

# === Plotting ===
def construct_accuracy_map(shape, year, county_class):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    plt.title(f'Urbanicity {county_class} – {year} RF Accuracy Map', size=16, weight='bold')

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
        inset.plot(ax=ax, column='PlotValue', cmap=cmap, edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})

    for ax in [main_ax, alaska_ax, hawaii_ax]:
        ax.axis('off')

    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

    # Colorbar
    bounds = np.linspace(0, 1, 21)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(bounds)
    cbar.ax.set_yticklabels([f'{int(b*100)}%' for b in bounds])
    cbar.set_label('Accuracy', fontsize=10, weight='bold')

    plt.savefig(f'{OUT_DIR}/Urbanicity_{county_class}_RF_Accuracy_Map_{year}.png', bbox_inches='tight', dpi=300)
    plt.close()

# === Run ===
def main():
    shape = load_shapefile()
    class_df = load_county_classes()
    county_classes = sorted(class_df['county_class'].unique())

    for county_class in county_classes:
        for year in range(2010, 2023):
            try:
                print(f"🗺️ Generating map for Category {county_class}, Year {year}...")
                preds = load_yearly_preds(year, county_class)
                merged = merge_data(shape.copy(), preds, county_class, class_df)
                construct_accuracy_map(merged, year, county_class)
            except FileNotFoundError:
                print(f"⚠️ Missing predictions for Category {county_class}, Year {year}, skipping.")

if __name__ == "__main__":
    main()
