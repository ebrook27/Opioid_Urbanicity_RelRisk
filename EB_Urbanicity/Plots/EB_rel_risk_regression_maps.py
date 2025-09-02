### 4/16/25, EB: I was able to get the Random Forest Regression model to work predicting relative risk scores.
### Its performance is ok, but doesn't seem great to me (but I'm used to incredibly tight errors from HAMMER project, so I could be wrong).
### Andrew suggested I try emulating his accuracy maps with my relative risk predictions, so that's what I'm trying to do here.
### I've saved each year's predictions to a .csv in the Regression_Preds folder. It's imported here and calculates the absolute error
### then plots the results for all counties onto a US map.

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
PRED_DIR = 'County Classification/Regression_Preds'
OUT_DIR = 'County Classification/Regression_Plots'
os.makedirs(OUT_DIR, exist_ok=True)

# === Load county shapefile ===
def load_shapefile():
    shape = gpd.read_file(SHAPE_PATH)
    shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)
    return shape

# === Load predictions for one year ===
def load_yearly_preds(year):
    file_path = f'{PRED_DIR}/{year}_MR_predictions.csv'
    df = pd.read_csv(file_path, dtype={'FIPS': str})
    df['FIPS'] = df['FIPS'].str.zfill(5)
    df['Absolute_Error'] = abs(df[f'{year}_True_MR'] - df[f'{year}_Pred_MR'])
    max_err = df['Absolute_Error'].max()
    df['Accuracy'] = 1 - (df['Absolute_Error'] / max_err)
    df['Accuracy'] = df['Accuracy'].clip(lower=0.0001, upper=0.9999)
    return df[['FIPS', 'Accuracy']]

# === Merge with shapefile ===
def merge_data(shape, acc_df):
    return shape.merge(acc_df, on='FIPS', how='left')

# === Plotting ===
def construct_accuracy_map(shape, year):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    plt.title(f'{year} LSTM Accuracy Map', size=16, weight='bold')

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

    # Colorbar
    bounds = np.linspace(0, 1, 21)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(bounds)
    cbar.ax.set_yticklabels([f'{int(b*100)}%' for b in bounds])
    cbar.set_label('Accuracy', fontsize=10, weight='bold')

    # plt.savefig(f'{OUT_DIR}/{year}_regression_accuracy_map.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUT_DIR}/{year}_mortality_regression_accuracy_map.png', bbox_inches='tight', dpi=300)
    plt.close()

# === Run ===
def main():
    shape = load_shapefile()
    for year in range(2013, 2022):
        print(f"🗺️ Generating map for {year}...")
        preds = load_yearly_preds(year)
        merged = merge_data(shape, preds)
        construct_accuracy_map(merged, year)
    print("✅ All maps saved.")

if __name__ == "__main__":
    main()
