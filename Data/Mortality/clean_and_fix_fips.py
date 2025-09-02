import pandas as pd
import geopandas as gpd
from tobler.area_weighted import area_interpolate

RAW_MORTALITY_NAMES = ['FIPS', 'Deaths', 'Population', 'Crude Rate']
COLUMNS_TO_KEEP = ['FIPS', 'Deaths', 'Population']
MISSING_VALUE = -9

def clean_rates(year):
    input_path = f'Data/Mortality/Raw Files/{year}_cdc_wonder_raw_mortality.csv'
    mort_df = pd.read_csv(input_path, header=0, names=RAW_MORTALITY_NAMES)
    
    # Format the columns
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df['Deaths'] = mort_df['Deaths'].astype(str) # need to be strings for now b/c they contain "suppressed/missing/etc" values
    mort_df['Population'] = mort_df['Population'].astype(str) # these also contain "suppressed/missing/etc" values so they need to be strings as well

    # Keep only the required columns
    mort_df = mort_df[COLUMNS_TO_KEEP]
    
    # Replace "Supressed/Missing/Not Available" values with -9
    mort_df['Deaths'] = mort_df['Deaths'].replace(['Suppressed', 'Missing', 'Not Available'], MISSING_VALUE)
    mort_df['Population'] = mort_df['Population'].replace(['Suppressed', 'Missing', 'Not Available'], MISSING_VALUE)
    
    # Convert 'Deaths' and 'Population' columns to float and int respectively
    mort_df['Deaths'] = mort_df['Deaths'].astype(float)
    mort_df['Population'] = mort_df['Population'].astype(int)
    
    # Create a new column for the mortality rates 
    mort_df[f'{year} MR'] = mort_df.apply(
        lambda row: MISSING_VALUE if row['Deaths'] == MISSING_VALUE or row['Population'] <= 0 else (row['Deaths'] / row['Population']) * 100000,
        axis=1
    )

    # Round the mortality rate column to 2 decimal places
    mort_df[f'{year} MR'] = mort_df[f'{year} MR'].round(2)
    return mort_df

def impute_old_ct_data(mort_df, year):
    neighs_path = 'Data/Neighbors/2020_neighbors_list.csv'
    neighs_names = ['FIPS', 'Neighbors']
    neighs_df = pd.read_csv(neighs_path, header=None, names=neighs_names)

    neighs_df['FIPS'] = neighs_df['FIPS'].astype(str).str.zfill(5)
    neighs_df['Neighbors'] = neighs_df['Neighbors'].apply(
        lambda x: x.split(',') if isinstance(x, str) and ',' in x else ([] if pd.isna(x) or x == '' else [x])
    )

    mort_df = mort_df.set_index('FIPS')
    for fips, row in mort_df.iterrows():
        if fips in ['09001', '09003', '09005', '09007', '09009', '09011', '09013', '09015']:
            if row[f'{year} MR'] == -9.0:
                neighbors = neighs_df.loc[neighs_df['FIPS'] == fips, 'Neighbors']
                neighbors = neighbors.values[0]
                available_neighbors = [neighbor for neighbor in neighbors if neighbor in mort_df.index and mort_df.loc[neighbor, f'{year} MR'] != -9]

                if len(available_neighbors) > 0:
                    new_value = sum([mort_df.loc[neighbor, f'{year} MR'] for neighbor in available_neighbors]) / len(available_neighbors)
                    mort_df.loc[fips, f'{year} MR'] = new_value
                else:
                    print("ERROR: A CT county is missing all neighbors.")
    mort_df = mort_df.reset_index()
    ct_df = mort_df[mort_df['FIPS'].str.startswith('09')]
    return ct_df

def fix_connecticut(mort_df, year):
    # Load 2020 and 2022 shapefiles for Connecticut
    # 2020 shapefile has the old county structure (matches will all years: 2010 - 2021)
    # 2022 shapefile has the new county structure
    old_shapefile_path = '2020 USA County Shapefile/Filtered Files/2020_filtered_shapefile.shp'
    new_shapefile_path = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
    
    old_ct_shape = gpd.read_file(old_shapefile_path)
    new_ct_shape = gpd.read_file(new_shapefile_path)

    # Filter shapefiles for Connecticut only (FIPS codes starting with '09')
    old_ct_shape = old_ct_shape[old_ct_shape['FIPS'].str.startswith('09')]
    new_ct_shape = new_ct_shape[new_ct_shape['FIPS'].str.startswith('09')]

    # Tobler needs a projected CRS (e.g., UTM) to run
    old_ct_shape = old_ct_shape.to_crs(epsg=26918)  # UTM zone 18N (for Connecticut)
    new_ct_shape = new_ct_shape.to_crs(epsg=26918)  # UTM zone 18N

    # Get the old CT data
    ct_df = impute_old_ct_data(mort_df, year)
    
    # Merge the old CT data with the old CT shape
    old_ct_shape = old_ct_shape.merge(ct_df, how='left', on='FIPS')
    
    # Perform area-weighted interpolation to the new county structure
    # We want to use extensive variables b/c mortality rates depend on population size
    interpolated_df = area_interpolate(source_df=old_ct_shape,
                                       target_df=new_ct_shape,
                                       extensive_variables=[f'{year} MR'])

    # Add FIPS column back the interpolated_df
    interpolated_df['FIPS'] = new_ct_shape['FIPS'].values
    
    # Merge interpolated data with the 2022 county shapefile
    new_ct_shape = new_ct_shape.merge(interpolated_df, how='left', on='FIPS')

    # Construct the fixed CT dataframe
    fixed_ct_df = new_ct_shape[['FIPS', f'{year} MR']].copy()
    fixed_ct_df[f'{year} MR'] = fixed_ct_df[f'{year} MR'].round(2)
    
    # Remove the old CT data from the original DataFrame
    mort_df = mort_df[~mort_df['FIPS'].str.startswith('09')]

    # Append the fixed CT data to the original DataFrame
    mort_df = pd.concat([mort_df, fixed_ct_df], ignore_index=True)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def load_shapefile():
    shapefile_path = f'2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
    shape = gpd.read_file(shapefile_path)
    return shape

def filter_fips_codes(year, mort_df, shape):
    # Extract FIPS codes from both DataFrames
    data_fips = mort_df['FIPS']
    shape_fips = shape['FIPS'] # there are 3144 counties in the shapefile

    # Keep only the counties in the data that are also in the shape 
    # (counties in data but not in shape won't show up on a map)
    filtered_df = mort_df[mort_df['FIPS'].isin(shape_fips)].reset_index(drop=True)

    # Counties in shape but not in data
    # (counties that will show up on the map but for which we don't have data)
    missing_data = shape_fips[~shape_fips.isin(data_fips)]

    # Add missing counties to the data with Mortality Rates set to -9
    missing_df = pd.DataFrame({'FIPS': missing_data, f'{year} MR': -9})
    filtered_df = pd.concat([filtered_df, missing_df], ignore_index=True)
    filtered_df = filtered_df.sort_values(by='FIPS').reset_index(drop=True)

    # Save the final filtered result
    output_path = f'Data/Mortality/Interim Files/{year}_mortality_interim.csv'
    filtered_df.to_csv(output_path, index=False)
    print(f'{year} interim file saved.')

def main():
    for year in range(2010,2023):
        mort_df = clean_rates(year)

        if year < 2022: # In 2022, the data for CT is genuinely missing
            mort_df = fix_connecticut(mort_df, year) 

        shape = load_shapefile()
        filter_fips_codes(year, mort_df, shape)

if __name__ == "__main__":
    main()