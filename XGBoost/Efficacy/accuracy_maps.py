import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} MR' for year in range(2010, 2023)]

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def load_mortality_rates(data_path, data_names):
    mort_df = pd.read_csv(data_path, header=0, names=data_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    mort_df[data_names[1:]] = mort_df[data_names[1:]].astype(float).clip(lower=0)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def load_yearly_predictions():
    preds_df = pd.DataFrame()
    for year in range(2011,2023):
        yearly_path = f'XGBoost/XGBoost Predictions/{year}_xgboost_predictions.csv'
        yearly_names = ['FIPS'] + [f'{year} Preds']
        yearly_df = pd.read_csv(yearly_path, header=0, names=yearly_names)
        yearly_df['FIPS'] = yearly_df['FIPS'].astype(str).str.zfill(5)
        yearly_df[f'{year} Preds'] = yearly_df[f'{year} Preds'].astype(float)

        if preds_df.empty:
            preds_df = yearly_df
        else:
            preds_df = pd.merge(preds_df, yearly_df, on='FIPS', how='outer')

    preds_df = preds_df.sort_values(by='FIPS').reset_index(drop=True)
    return preds_df

def calculate_accuracy(mort_df, preds_df, year):
    acc_df = mort_df[['FIPS']].copy()
    acc_df[f'{year} Absolute Errors'] = abs(preds_df[f'{year} Preds'] - mort_df[f'{year} MR'])
    max_abs_err = acc_df[f'{year} Absolute Errors'].max()

   # Accuracy calculation
    if max_abs_err == 0: # Perfect match
        # Assign slightly less than 1 to remain in cmap interval
        acc_df[f'{year} Accuracy'] = 0.9999 
    else:
        # Calculate accuracy as normal
        acc_df[f'{year} Accuracy'] = 1 - (acc_df[f'{year} Absolute Errors'] / max_abs_err)

        # Then adjust accuracy to 0.9999 if it's exactly 1, and to 0.0001 if it's exactly 0, to remain in cmap interval
        acc_df[f'{year} Accuracy'] = acc_df[f'{year} Accuracy'].apply(lambda x: 0.9999 if x == 1 else (0.0001 if x == 0 else x))
    return acc_df

def merge_data_shape(shape, acc_df):
    return shape.merge(acc_df, on='FIPS')

def construct_accuracy_map(shape, year):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    title = f'{year} XGBoost Accuracy Map'
    plt.title(title, size=16, weight='bold')

    # Alaska and Hawaii insets
    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4]) 
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])  
    
    # Plot state boundaries
    state_boundaries = shape.dissolve(by='STATEFP', as_index=False)
    state_boundaries.boundary.plot(ax=main_ax, edgecolor='black', linewidth=.5)

    alaska_state = state_boundaries[state_boundaries['STATEFP'] == '02']
    alaska_state.boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=.5)

    hawaii_state = state_boundaries[state_boundaries['STATEFP'] == '15']
    hawaii_state.boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=.5)

    # Cmap for accuracy
    num_intervals = 20
    cmap = plt.get_cmap('RdYlGn', num_intervals)

    # Define the insets for coloring
    shapes = [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax, 'continental'),
        (shape[shape['STATEFP'] == '02'], alaska_ax, 'alaska'),
        (shape[shape['STATEFP'] == '15'], hawaii_ax, 'hawaii') ]

    # Color the maps
    for inset, ax, _ in shapes:
        for _, row in inset.iterrows():
            county = row['FIPS']
            acc = row[f'{year} Accuracy']
            color = cmap(acc)
            inset[inset['FIPS'] == county].plot(ax=ax, color=color)

    # Plot county boundaries with thin black lines
    shape.boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.1)
    shape[shape['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.1)
    shape[shape['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.1)

    # Adjust the viewing
    set_view_window(main_ax,alaska_ax,hawaii_ax)

    # Add the colorbar
    add_colorbar(main_ax, cmap)

    plt.savefig(f'XGBoost/Efficacy/Accuracy Maps/{year}_xgb_acc_map', bbox_inches=None, pad_inches=0, dpi=300)
    # plt.show()
    plt.close(fig)

def set_view_window(main_ax,alaska_ax,hawaii_ax):
    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)
    alaska_ax.set_axis_off()
    hawaii_ax.set_axis_off()
    main_ax.axis('off')

    # Fix window
    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

def add_colorbar(main_ax, cmap):
    # Accuracy levels
    color_bounds = np.linspace(0, 100, 21)  # 5% intervals
    norm = BoundaryNorm(color_bounds, cmap.N)
    cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(color_bounds)
    cbar.set_ticklabels([f'{i}%' for i in color_bounds])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Accuracy Levels', fontsize=10, weight='bold')

def main():
    shape = load_shapefile(SHAPE_PATH)
    mort_df = load_mortality_rates(MORTALITY_PATH, MORTALITY_NAMES)
    preds_df = load_yearly_predictions()
    
    for year in range(2011, 2023):
        acc_df = calculate_accuracy(mort_df, preds_df, year)
        shape = merge_data_shape(shape, acc_df)
        construct_accuracy_map(shape, year)
        print(f'Plot printed for {year}.')

if __name__ == "__main__":
    main()