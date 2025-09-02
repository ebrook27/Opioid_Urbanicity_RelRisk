import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")

def load_shapefile():
    shape_path = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
    shape = gpd.read_file(shape_path)
    shape['FIPS'] = shape['FIPS'].astype(str)
    return shape

def load_final_data():
    mort_path = f'Data/Mortality/Final Files/Mortality_final_rates.csv'
    mort_names = ['FIPS'] + [f'{year} MR' for year in range(2010, 2023)]
    mort_df = pd.read_csv(mort_path, header=0, names=mort_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[mort_names[1:]] = mort_df[mort_names[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def merge_data_shape(shape, mort_df):
    shape = shape.merge(mort_df, on='FIPS')
    return shape

def custom_colormap():
    """
    Construct a custom RdYlBu colormap using colorbrewer to ensure a colorblind-friendly color 
    scheme.
    """

    # Define the CSS RdYlBu color scheme manually (css copied from colorbrewer)
    css_RdYlBu_colors = [
        (165/255, 0/255, 38/255),    # Dark Red
        (215/255, 48/255, 39/255),
        (244/255, 109/255, 67/255),
        (253/255, 174/255, 97/255),
        (254/255, 224/255, 144/255), # Light Orange
        (255/255, 255/255, 191/255), # Yellow (Neutral)
        (224/255, 243/255, 248/255), # Light Blue
        (171/255, 217/255, 233/255),
        (116/255, 173/255, 209/255),
        (69/255, 117/255, 180/255),
        (49/255, 54/255, 149/255)    # Dark Blue
    ]

    # Create the custom colormap
    custom_RdYlBu = mcolors.LinearSegmentedColormap.from_list("customRdYlBu", css_RdYlBu_colors, N=256)
    cmap = custom_RdYlBu.reversed()  # Reverse so that blue is low and red is high
    return cmap

def plot_heat_map(shape, year):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    title = f'{year} Final Heat Map for the Mortality Rates'
    plt.title(title, size=13, weight='bold')

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

    yearly_data = shape[f'{year} MR'].values

    # Compute the empirical percentiles for each value
    percentiles = np.percentile(yearly_data, np.arange(0, 101, 1))

    # Color the maps
    cmap = custom_colormap()
  
    for inset, ax, _ in shapes:
        for _, row in inset.iterrows():
            county = row['FIPS']
            data_value = row[f'{year} MR']
            
            # Calculate the empirical percentile for the data_value
            percentile_rank = np.sum(data_value > percentiles) / 100
            
            # Map the percentile (0 to 1) to a color
            color = cmap(percentile_rank)
            
            inset[inset['FIPS'] == county].plot(ax=ax, color=color)

    # Plot county boundaries with thin black lines
    shape.boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.1)
    shape[shape['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.1)
    shape[shape['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.1)

    # Adjust the viewing window
    set_view_window(main_ax,alaska_ax,hawaii_ax)

    # Add the colorbar
    add_color_bar(main_ax)

    # Save the map
    output_map_path = f'Missing Data Results/Mortality Maps/Final/{year}_final_mort_heat_map.png'
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

def add_color_bar(main_ax):
    # Get the colormap
    cmap = custom_colormap()
    
    # Define color bounds and normalization
    color_bounds = np.linspace(0, 1, 21)  # 21 points for 0%, 5%, ..., 100%
    norm = BoundaryNorm(color_bounds, ncolors=cmap.N, clip=True)
    
    # Create the colorbar
    cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    # Set tick positions and labels
    tick_positions = np.linspace(0, 1, 21)  # 21 points for 0%, 5%, ..., 100%
    cbar.set_ticks(tick_positions)
    label_list = [f'{i}%' for i in range(0, 101, 5)]
    cbar.set_ticklabels(label_list)
    
    # Customize the colorbar
    cbar.ax.tick_params(axis='y', labelsize=8) 
    cbar.set_label('Percentiles', fontsize=10, weight='bold')

def main():
    shape = load_shapefile()
    mort_df = load_final_data()
    shape = merge_data_shape(shape, mort_df)
    
    for year in range(2010, 2023):
        plot_heat_map(shape, year)
        print(f'Plot printed for {year}.')

if __name__ == "__main__":
    main()