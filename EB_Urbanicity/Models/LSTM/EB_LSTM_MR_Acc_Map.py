### 6/18/25, EB: In the file EB_LSTM_Mortality_Model.py, I used an LSTM model to predict mortality rates for several years in the study. The error histograms seem to indicate that the model is 
### performing well, but I want to visualize the accuracy of the predictions on a map.

import pandas as pd
import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── CONFIG ──────────────────────────────────────────────────────────────────
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
LSTM_PRED_CSV = 'County Classification/LSTM_MR_Preds/LSTM_MR_predictions_threeyear.csv'                 # ← your predictions
OUT_DIR = 'County Classification/LSTM_MR_Acc_Maps'
os.makedirs(OUT_DIR, exist_ok=True)

# ── HELPERS ─────────────────────────────────────────────────────────────────
def load_shapefile():
    shp = gpd.read_file(SHAPE_PATH)
    shp['FIPS'] = shp['FIPS'].astype(str).str.zfill(5)
    return shp

def load_lstm_preds():
    df = pd.read_csv(LSTM_PRED_CSV, dtype={'FIPS': str})
    df['FIPS'] = df['FIPS'].str.zfill(5)
    df['Absolute_Error'] = np.abs(df['Pred_MR'] - df['True_MR'])
    return df

def accuracy_by_year(df, year):
    df_year = df[df['Year'] == year].copy()
    max_err = df_year['Absolute_Error'].max()
    df_year['Accuracy'] = 1 - (df_year['Absolute_Error'] / max_err)
    df_year['Accuracy'] = df_year['Accuracy'].clip(lower=0.0001, upper=0.9999)
    return df_year[['FIPS', 'Accuracy']]

def merge_data(shape, acc_df):
    return shape.merge(acc_df, on='FIPS', how='left')

def plot_accuracy_map(shape, year):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    plt.title(f'{year} LSTM Accuracy Map', size=16, weight='bold')

    # ─ insets for AK & HI ─
    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])

    state_bounds = shape.dissolve(by='STATEFP', as_index=False)
    state_bounds.boundary.plot(ax=main_ax, edgecolor='black', linewidth=.5)
    state_bounds[state_bounds['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=.5)
    state_bounds[state_bounds['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=.5)

    cmap = plt.get_cmap('RdYlGn', 20)
    for inset, ax in [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax),
        (shape[shape['STATEFP'] == '02'], alaska_ax),
        (shape[shape['STATEFP'] == '15'], hawaii_ax)
    ]:
        inset.plot(ax=ax, column='Accuracy', cmap=cmap, edgecolor='black', linewidth=0.1)

    for ax in (main_ax, alaska_ax, hawaii_ax):
        ax.axis('off')

    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

    # colorbar
    bounds = np.linspace(0, 1, 21)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(bounds)
    cbar.ax.set_yticklabels([f'{int(b*100)}%' for b in bounds])
    cbar.set_label('Accuracy', fontsize=10, weight='bold')

    out_path = os.path.join(OUT_DIR, f'LSTM_threeyear_Accuracy_Map_{year}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_path}")

# ── MAIN PIPELINE ────────────────────────────────────────────────────────────
def main():
    shape = load_shapefile()
    preds = load_lstm_preds()
    years = sorted(preds['Year'].unique())

    for yr in years:
        print(f"🗺️ Generating accuracy map for {yr}...")
        acc_df = accuracy_by_year(preds, yr)
        merged = merge_data(shape.copy(), acc_df)
        plot_accuracy_map(merged, yr)

    print("✅ All LSTM accuracy maps saved.")

# ── run ─
if __name__ == "__main__":
    main()
