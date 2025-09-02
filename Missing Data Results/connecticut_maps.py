import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import rgb_to_hsv
import warnings
warnings.filterwarnings("ignore")

# Constants
OLD_CT_FIPS = ['09001', '09003', '09005', '09007', '09009', '09011', '09013', '09015']
NEW_CT_FIPS = ['09110', '09120', '09130', '09140', '09150', '09160', '09170', '09180', '09190']

def load_2020_shapefile():
    shape_2020_path = '2020 USA County Shapefile/Filtered Files/2020_filtered_shapefile.shp'
    shape_2020 = gpd.read_file(shape_2020_path)
    shape_2020['FIPS'] = shape_2020['FIPS'].astype(str)

    shape_2020 = shape_2020[shape_2020['FIPS'].isin(OLD_CT_FIPS)]
    return shape_2020

def load_2022_shapefile():
    shape_2022_path = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
    shape_2022 = gpd.read_file(shape_2022_path)
    shape_2022['FIPS'] = shape_2022['FIPS'].astype(str)

    shape_2022 = shape_2022[shape_2022['FIPS'].isin(NEW_CT_FIPS)]
    return shape_2022

def impute_old_ct_data(raw_df, year):
    neighs_path = 'Data/Neighbors/2020_neighbors_list.csv'
    neighs_names = ['FIPS', 'Neighbors']
    neighs_df = pd.read_csv(neighs_path, header=None, names=neighs_names)

    neighs_df['FIPS'] = neighs_df['FIPS'].astype(str).str.zfill(5)
    neighs_df['Neighbors'] = neighs_df['Neighbors'].apply(
        lambda x: x.split(',') if isinstance(x, str) and ',' in x else ([] if pd.isna(x) or x == '' else [x])
    )

    raw_df = raw_df.set_index('FIPS')
    for fips, row in raw_df.iterrows():
        if fips in ['09001', '09003', '09005', '09007', '09009', '09011', '09013', '09015']:
            if row[f'{year} MR'] == -9.0:
                neighbors = neighs_df.loc[neighs_df['FIPS'] == fips, 'Neighbors']
                neighbors = neighbors.values[0]
                available_neighbors = [neighbor for neighbor in neighbors if neighbor in raw_df.index and raw_df.loc[neighbor, f'{year} MR'] != -9]

                if len(available_neighbors) > 0:
                    new_value = sum([raw_df.loc[neighbor, f'{year} MR'] for neighbor in available_neighbors]) / len(available_neighbors)
                    raw_df.loc[fips, f'{year} MR'] = new_value
                else:
                    print("ERROR: A CT county is missing all neighbors.")
    raw_df = raw_df.reset_index()
    raw_df = raw_df[raw_df['FIPS'].str.startswith('09')]
    raw_df = raw_df.sort_values(by='FIPS').reset_index(drop=True)
    return raw_df

def load_raw_data(year):
    raw_path = f'Data/Mortality/Raw Files/{year}_cdc_wonder_raw_mortality.csv'
    raw_names = ['FIPS', f'{year} Deaths', f'{year} Pop', f'{year} MR']
    raw_df = pd.read_csv(raw_path, header=0, names=raw_names)

    raw_df['FIPS'] = raw_df['FIPS'].astype(str).str.zfill(5) 
    raw_df = raw_df[['FIPS', f'{year} MR']]

    raw_df[f'{year} MR'] = pd.to_numeric(raw_df[f'{year} MR'], errors='coerce').fillna(0) # if data is missing, we don't use it to impute
    raw_df = impute_old_ct_data(raw_df, year)
    return raw_df

def load_final_data(year):
    final_path = f'Data/Mortality/Final Files/Mortality_final_rates.csv'
    final_names = ['FIPS'] + [f'{year} MR' for year in range(2010, 2023)]
    final_df = pd.read_csv(final_path, header=0, names=final_names)
    final_df['FIPS'] = final_df['FIPS'].astype(str).str.zfill(5)
    final_df[final_names[1:]] = final_df[final_names[1:]].astype(float)

    final_df = final_df[final_df['FIPS'].isin(NEW_CT_FIPS)][['FIPS', f'{year} MR']]
    final_df = final_df.sort_values(by='FIPS').reset_index(drop=True)
    return final_df

def merge_shapes_with_data(shape_2020, shape_2022, final_df, raw_df):
    shape_2020 = shape_2020.merge(raw_df, on='FIPS')
    shape_2022 = shape_2022.merge(final_df, on='FIPS')
    return shape_2020, shape_2022

def plot_comparison_heat_map(shape_2020, shape_2022, year):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    title = f'Results of Connecticut Data Interpolation in {year}'
    plt.suptitle(title, size=15, weight='bold')

    old_data = shape_2020[f'{year} MR'].values
    new_data = shape_2022[f'{year} MR'].values

    cmap = plt.get_cmap('RdYlBu_r')

    # Create a common color normalization for both maps
    vmin = min(np.min(old_data), np.min(new_data))
    vmax = max(np.max(old_data), np.max(new_data))
    norm = BoundaryNorm(np.linspace(vmin, vmax, 21), cmap.N)

    # Function to decide the text color based on background color
    def get_text_color(value, cmap, norm):
        rgba = cmap(norm(value))  # Get the RGBA color for the value
        hsv = rgb_to_hsv(rgba[:3])  # Convert to HSV to evaluate brightness
        return 'white' if hsv[2] < 0.65 else 'black'  # Use white if value is dark

    # Plot the old data map
    shape_2020.plot(column=f'{year} MR', cmap=cmap, linewidth=0.8, ax=axes[0], edgecolor='0.8', norm=norm)
    axes[0].set_title(f'Old CT County Structure', fontsize=12)
    axes[0].axis('off')

    # Add FIPS codes to the old CT map with dynamic text color
    for idx, row in shape_2020.iterrows():
        centroid = row['geometry'].centroid
        mr_value = row[f'{year} MR']
        text_color = get_text_color(mr_value, cmap, norm)
        axes[0].annotate(text=row['FIPS'], xy=(centroid.x, centroid.y), ha='center', fontsize=8, color=text_color)

    # Plot the new data map
    shape_2022.plot(column=f'{year} MR', cmap=cmap, linewidth=0.8, ax=axes[1], edgecolor='0.8', norm=norm)
    axes[1].set_title(f'New CT County Structure', fontsize=12)
    axes[1].axis('off')

    # Add FIPS codes to the new CT map with dynamic text color
    for idx, row in shape_2022.iterrows():
        centroid = row['geometry'].centroid
        mr_value = row[f'{year} MR']
        text_color = get_text_color(mr_value, cmap, norm)
        axes[1].annotate(text=row['FIPS'], xy=(centroid.x, centroid.y), ha='center', fontsize=8, color=text_color)

    # Add colorbar 
    sm = ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)

    # Add label to the colorbar
    cbar.set_label('Mortality Rates', fontsize=12, weight='bold')

    # Customize tick labels 
    tick_values = np.linspace(vmin, vmax, 5)  # Adjust number of ticks if needed
    tick_labels = [f'{val:.2f}' for val in tick_values]
    tick_labels[-1] = f'{vmax:.2f}'  # Add "Max: " to the final tick label
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)

    # Save the plot
    output_map_path = f'Missing Data Results/Connecticut/{year}_ct_comparison.png'
    plt.savefig(output_map_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

def main():
    for year in range(2010,2022):
        shape_2020 = load_2020_shapefile()
        shape_2022 = load_2022_shapefile()

        raw_df = load_raw_data(year)
        final_df = load_final_data(year)

        shape_2020, shape_2022 = merge_shapes_with_data(shape_2020, shape_2022, final_df, raw_df)

        plot_comparison_heat_map(shape_2020, shape_2022, year)
        print(f'Plot printed for {year}.')

if __name__ == "__main__":
    main()