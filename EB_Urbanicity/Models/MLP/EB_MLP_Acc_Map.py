### 6/12/25, EB: In the file EB_MLP_Mortality_Model.py, I used a MLP model to predict mortality rates for the year 2021.
### using the SVI data for all years prior. I'm having trouble also predicting mortality for 2022, but I want to 
### look at the accuracy of the 2021 predictions.

import pandas as pd
import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

# === PATHS ===
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
MLP_PRED_PATH = 'County Classification/MLP_Results/PyTorch_MLP_model_preds_13.7.3.1.csv'
OUT_DIR = 'County Classification/MLP_Results/Plots/MLP_Accuracy_Maps'
os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD SHAPEFILE ===
def load_shapefile():
    shape = gpd.read_file(SHAPE_PATH)
    shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)
    return shape

# === LOAD MLP PREDICTIONS ===
def load_mlp_preds(year=2021):
    df = pd.read_csv(MLP_PRED_PATH, dtype={'FIPS': str})
    df = df[df['year'] == year].copy()
    df['FIPS'] = df['FIPS'].str.zfill(5)
    max_err = df['Absolute_Error'].max()
    df['Accuracy'] = 1 - (df['Absolute_Error'] / max_err)
    df['Accuracy'] = df['Accuracy'].clip(lower=0.0001, upper=0.9999)
    return df[['FIPS', 'Accuracy']]

# === MERGE SHAPE + DATA ===
def merge_data(shape, acc_df):
    return shape.merge(acc_df, on='FIPS', how='left')

# === PLOT MAP ===
def construct_accuracy_map(shape, year=2021):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    plt.title(f'{year} MLP Accuracy Map', size=16, weight='bold')

    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])

    state_boundaries = shape.dissolve(by='STATEFP', as_index=False)
    state_boundaries.boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.5)
    state_boundaries[state_boundaries['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.5)
    state_boundaries[state_boundaries['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.5)

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

    plt.savefig(f'{OUT_DIR}/MLP_Accuracy_Map_{year}_13.7.3.1.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f'✅ Saved accuracy map for {year} to: {OUT_DIR}/MLP_Accuracy_Map_{year}_13.7.3.1.png')

# === PIPELINE ===
def run_mlp_accuracy_map(year=2021):
    shape = load_shapefile()
    preds = load_mlp_preds(year)
    merged = merge_data(shape, preds)
    construct_accuracy_map(merged, year)

# === CALL THE FUNCTION FOR 2021 ===
if __name__ == '__main__':
    run_mlp_accuracy_map(year=2021)
