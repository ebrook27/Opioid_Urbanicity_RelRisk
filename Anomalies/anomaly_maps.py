import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from scipy.stats import lognorm

# Constants
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]
TAIL = 1

def construct_output_map_path(year):
    output_map_path = f'Anomalies/Anomaly Maps/{TAIL}% Tails/{year}_{TAIL}%_tail_anomaly_map'
    return output_map_path

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def load_mort_rates():
    mort_df = pd.DataFrame()
    mort_df = pd.read_csv(MORTALITY_PATH, header=0, names=MORTALITY_NAMES)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[MORTALITY_NAMES[1:]] = mort_df[MORTALITY_NAMES[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def merge_data_shape(shape, mort_df):
    return shape.merge(mort_df, on='FIPS')

def construct_anomaly_map(shape, year, output_map_path):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    title = f'{year} Anomaly Map for the Mortality Rates: {TAIL}% Tails'
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

    # Define the insets for coloring
    shapes = [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax, 'continental'),
        (shape[shape['STATEFP'] == '02'], alaska_ax, 'alaska'),
        (shape[shape['STATEFP'] == '15'], hawaii_ax, 'hawaii') ]

    # Color the maps
    mort_rates = shape[f'{year} Mortality Rates'].values
    non_zero_mort_rates = mort_rates[mort_rates > 0]
    norm_params = lognorm.fit(non_zero_mort_rates)
    log_shape, loc, scale = norm_params

    tail = TAIL / 100
    upper_threshold = lognorm.ppf(1-tail, log_shape, loc, scale)
    lower_threshold = lognorm.ppf(tail, log_shape, loc, scale)
    print(f'Upper threshold = {upper_threshold} and lower threshold = {lower_threshold}')

    for inset, ax, _ in shapes:
        for _, row in inset.iterrows():
            county = row['FIPS']
            rate = row[f'{year} Mortality Rates']
            
            if rate > upper_threshold:
                color = 'red'
            elif 0 < rate < lower_threshold:
                color = 'blue'
            else:
                color = 'lightgrey'

            inset[inset['FIPS'] == county].plot(ax=ax, color=color)

    # Adjust the viewing
    set_view_window(main_ax,alaska_ax,hawaii_ax)

    # Add the colorbar
    add_legend(main_ax)

    plt.savefig(output_map_path, bbox_inches=None, pad_inches=0, dpi=300)
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

def add_legend(main_ax):
    red_patch = mpatches.Patch(color='red', label='Hot Anomaly')
    blue_patch = mpatches.Patch(color='blue', label='Cold Anomaly')
    main_ax.legend(handles=[red_patch, blue_patch], loc='lower right', bbox_to_anchor=(1.05, 0))

def main():
    for year in range(2010, 2023):
        output_map_path = construct_output_map_path(year)
        shape = load_shapefile(SHAPE_PATH)
        mort_df = load_mort_rates()
        shape = merge_data_shape(shape, mort_df)
        construct_anomaly_map(shape, year, output_map_path)
        print(f'Plot printed for {year}.')

if __name__ == "__main__":
    main()