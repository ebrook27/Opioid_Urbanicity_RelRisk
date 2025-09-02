import pandas as pd
import geopandas as gpd
from tobler.area_weighted import area_interpolate

# Constants
SVI_VAR_LIST = {
    'EPL_POV': 'Below Poverty',
    'EPL_UNEMP': 'Unemployment',
    'EPL_NOHSDP': 'No High School Diploma',
    'EPL_AGE65': 'Aged 65 or Older',
    'EPL_AGE17': 'Aged 17 or Younger',
    # 'EPL_DISABL': 'Disability',
    'EPL_SNGPNT': 'Single-Parent Household',
    'EPL_MINRTY': 'Minority Status',
    'EPL_LIMENG': 'Limited English Ability',
    'EPL_MUNIT': 'Multi-Unit Structures',
    'EPL_MOBILE': 'Mobile Homes',
    'EPL_CROWD': 'Crowding',
    'EPL_NOVEH': 'No Vehicle',
    'EPL_GROUPQ': 'Group Quarters'
}
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
PATH_2010 = 'Data/SVI/Raw Files/SVI_2010_US_county.csv'
PATH_2014 = 'Data/SVI/Raw Files/SVI_2014_US_county.csv'
PATH_2016 = 'Data/SVI/Raw Files/SVI_2016_US_county.csv'
PATH_2018 = 'Data/SVI/Raw Files/SVI_2018_US_county.csv'
PATH_2020 = 'Data/SVI/Raw Files/SVI_2020_US_county.csv'
PATH_2022 = 'Data/SVI/Raw Files/SVI_2022_US_county.csv'
MISSING_DATA_VALUE = -9

def load_dataframes():
    # Load 2010
    # 2010 had a different labeling scheme that we need to account for
    df_2010 = pd.read_csv(PATH_2010)
    df_2010['FIPS'] = df_2010['FIPS'].astype(str).str.zfill(5)  # Convert FIPS to strings and zero-pad from the left
    for var, _ in SVI_VAR_LIST.items():
        incorrect_name = var.replace('EPL', 'E_PL') # create the 2010 name from the correct name

        # Some variables we need to change more specifically
        if var == 'EPL_NOHSDP':
            incorrect_name = 'E_PL_NOHSDIP'
        if var == 'EPL_AGE65':
            incorrect_name = 'PL_AGE65' # No 'estimates' for this in 2010, just exact percentiles
        if var == 'EPL_AGE17':
            incorrect_name = 'PL_AGE17' # No 'estimates' for this in 2010, just exact percentiles
        if var == 'EPL_SNGPNT':
            incorrect_name = 'PL_SNGPRNT' # No 'estimates' for this in 2010, just exact percentiles, also acronym change
        if var == 'EPL_MINRTY':
            incorrect_name = 'PL_MINORITY' # No 'estimates' for this in 2010, just exact percentiles, also acronym change
        if var == 'EPL_GROUPQ':
            incorrect_name = 'PL_GROUPQ' # No 'estimates' for this in 2010, just exact percentiles

        df_2010.rename(columns={incorrect_name: var}, inplace=True)

    # Load 2014
    df_2014 = pd.read_csv(PATH_2014)
    df_2014['FIPS'] = df_2014['FIPS'].astype(str).str.zfill(5)

    # Load 2016
    df_2016 = pd.read_csv(PATH_2016)
    df_2016['FIPS'] = df_2016['FIPS'].astype(str).str.zfill(5)

    # Load 2018
    df_2018 = pd.read_csv(PATH_2018)
    df_2018['FIPS'] = df_2018['FIPS'].astype(str).str.zfill(5)

    # Load 2020
    df_2020 = pd.read_csv(PATH_2020)
    df_2020['FIPS'] = df_2020['FIPS'].astype(str).str.zfill(5)
    df_2020.rename(columns={'EPL_POV150': 'EPL_POV'}, inplace=True) # variable name changed to POV150 in 2020

    # Load 2022
    df_2022 = pd.read_csv(PATH_2022)
    df_2022['FIPS'] = df_2022['FIPS'].astype(str).str.zfill(5)
    df_2022.rename(columns={'EPL_POV150': 'EPL_POV'}, inplace=True)

    return df_2010, df_2014, df_2016, df_2018, df_2020, df_2022

def construct_var_dataframe(df_2010, df_2014, df_2016, df_2018, df_2020, df_2022, var):
    # Start with 2010 data
    variable_df = df_2010[['FIPS', var]].copy()  # Copy the FIPS and variable columns
    variable_df.rename(columns={var: f'2010 {var}'}, inplace=True)  # Account for the year

    # Merge the data for the other years one year at a time
    # Use outer joins to accumulate all FIPS codes across all years
    # Missing values for the non-existing years will be set to NaN by default
    # For example, in 2022 when the CT county structure changes, 
    # all these new counties will have NaN values for the years 2010 - 2021, 
    # but this is ok b/c we fix that in the next step
    data = [(df_2014, '2014'), (df_2016, '2016'), (df_2018, '2018'), (df_2020, '2020'), (df_2022, '2022')]
    
    for df, year in data:
        dummy_df = df[['FIPS', var]].copy()
        dummy_df.rename(columns={var: f'{year} {var}'}, inplace=True)

        # on='FIPS': specifies that the merge is based on matching FIPS codes across both variable_df (left) and dummy_df (right)
        # how='outer': ensures that all FIPS codes from both variable_df and dummy_df will be included in the result. 
        # If a FIPS code is missing in either dataframe, the corresponding values from that dataframe will be set to NaN.
        variable_df = pd.merge(variable_df, dummy_df, on='FIPS', how='outer')

    # Set the accumulated NaN values to the missing data value
    variable_df.fillna(-9, inplace=True)

    # Set the FIPS codes to be string values
    variable_df['FIPS'] = variable_df['FIPS'].astype(str).str.zfill(5) 

    return variable_df


def impute_data(variable_df, var):
    # Impute data for the missing years 
    # Skip calculation if any bordering year has missing data as we cannot impute properly in such cases
    def impute_years(start_year, end_year, mid_years):
        start_col = f'{start_year} {var}'
        end_col = f'{end_year} {var}'
        for year, weight in mid_years:
            mid_col = f'{year} {var}'
            variable_df[mid_col] = variable_df.apply(
                lambda row: row[start_col] + weight * (row[end_col] - row[start_col])
                if row[start_col] != -9.0 and row[end_col] != -9.0 else -9.0,
                axis=1
            )

    # Impute data for the missing years
    impute_years(2010, 2014, [(2011, 0.25), (2012, 0.50), (2013, 0.75)])
    impute_years(2014, 2016, [(2015, 0.50)])
    impute_years(2016, 2018, [(2017, 0.50)])
    impute_years(2018, 2020, [(2019, 0.50)])
    impute_years(2020, 2022, [(2021, 0.50)])

    return variable_df

def impute_old_ct_data(variable_df, var, year, old_ct_fips):
    neighs_path = 'Data/Neighbors/2020_neighbors_list.csv'
    neighs_names = ['FIPS', 'Neighbors']
    neighs_df = pd.read_csv(neighs_path, header=None, names=neighs_names)

    neighs_df['FIPS'] = neighs_df['FIPS'].astype(str).str.zfill(5)
    neighs_df['Neighbors'] = neighs_df['Neighbors'].apply(
        lambda x: x.split(',') if isinstance(x, str) and ',' in x else ([] if pd.isna(x) or x == '' else [x])
    )

    variable_df = variable_df.set_index('FIPS')
    for fips, row in variable_df.iterrows():
        if fips in old_ct_fips:
            if row[f'{year} {var}'] == -9.0:
                neighbors = neighs_df.loc[neighs_df['FIPS'] == fips, 'Neighbors']
                neighbors = neighbors.values[0]
                available_neighbors = [neighbor for neighbor in neighbors if neighbor in variable_df.index and variable_df.loc[neighbor, f'{year} {var}'] != -9]

                if len(available_neighbors) > 0:
                    new_value = sum([variable_df.loc[neighbor, f'{year} {var}'] for neighbor in available_neighbors]) / len(available_neighbors)
                    variable_df.loc[fips, f'{year} {var}'] = new_value
                else:
                    print("ERROR: A CT county is missing all neighbors.")
    variable_df = variable_df.reset_index()
    yearly_ct_df = variable_df[variable_df['FIPS'].isin(old_ct_fips)][['FIPS', f'{year} {var}']]
    return yearly_ct_df

def fix_connecticut(variable_df, var):
    for year in range(2010, 2022):
        old_ct_fips = ['09001', '09003', '09005', '09007', '09009', '09011', '09013', '09015']
        yearly_ct_df = impute_old_ct_data(variable_df, var, year, old_ct_fips)

        old_shapefile_path = '2020 USA County Shapefile/Filtered Files/2020_filtered_shapefile.shp'
        new_shapefile_path = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
        
        old_ct_shape = gpd.read_file(old_shapefile_path)
        new_ct_shape = gpd.read_file(new_shapefile_path)

        # Filter shapefiles for Connecticut only (FIPS codes starting with '09')
        old_ct_shape = old_ct_shape[old_ct_shape['FIPS'].isin(old_ct_fips)]
        new_ct_shape = new_ct_shape[new_ct_shape['FIPS'].str.startswith('09')]

        # Tobler needs a projected CRS (e.g., UTM) to run
        old_ct_shape = old_ct_shape.to_crs(epsg=26918)  # UTM zone 18N (for Connecticut)
        new_ct_shape = new_ct_shape.to_crs(epsg=26918)  # UTM zone 18N

        # Merge the old CT data with the old CT shape
        old_ct_shape = old_ct_shape.merge(yearly_ct_df, how='left', on='FIPS')

        # Perform area-weighted interpolation to the new county structure
        interpolated_df = area_interpolate(source_df=old_ct_shape,
                                           target_df=new_ct_shape,
                                           extensive_variables=[f'{year} {var}'])

        # Add FIPS column back to the interpolated_df
        interpolated_df['FIPS'] = new_ct_shape['FIPS'].values
        
        # Merge interpolated data with the 2022 county shapefile
        new_ct_shape_year = new_ct_shape.merge(interpolated_df, how='left', on='FIPS')

        # Update the CT rows in the variable_df with the imputed data
        variable_df = variable_df.set_index('FIPS')
        fixed_ct_df = new_ct_shape_year[['FIPS', f'{year} {var}']].set_index('FIPS')
        variable_df.update(fixed_ct_df) # at the CT FIPS indices and in the specified column

        # Reset the index to make FIPS a column again
        variable_df = variable_df.reset_index()

    # Need to fix the imputation for 2021
    # These values were previously imputed from the old county structure to the new, making them nonsense
    # But we can fix it now that the new county structure has been accounted for in each previous year
    new_ct_fips = ['09110', '09120', '09130', '09140', '09150', '09160', '09170', '09180', '09190']
    for fips in new_ct_fips:
        prev_value = variable_df[variable_df['FIPS'] == fips][f'2020 {var}'].values[0]
        next_value = variable_df[variable_df['FIPS'] == fips][f'2022 {var}'].values[0]

        value_2021 = prev_value + 0.50 * (next_value - prev_value)
        variable_df.loc[variable_df['FIPS'] == fips, f'2021 {var}'] = value_2021

    # Final sort and reset index
    variable_df = variable_df.sort_values(by='FIPS').reset_index(drop=True)
    return variable_df

def fix_rio_arriba(variable_df, var):
    # Fix the data error for Rio Arriba NM in 2018
    variable_df.loc[variable_df['FIPS'] == '35039', f'2017 {var}'] = variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] + 0.25 * ( variable_df.loc[variable_df['FIPS'] == '35039', f'2020 {var}'] - variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] )
    variable_df.loc[variable_df['FIPS'] == '35039', f'2018 {var}'] = variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] + 0.50 * ( variable_df.loc[variable_df['FIPS'] == '35039', f'2020 {var}'] - variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] )
    variable_df.loc[variable_df['FIPS'] == '35039', f'2019 {var}'] = variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] + 0.75 * ( variable_df.loc[variable_df['FIPS'] == '35039', f'2020 {var}'] - variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] )
    return variable_df

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)
    return shape

def fix_fips(shape, variable_df, var):
    # Keep only counties that exist in the shape
    # Remember that FIPS codes were accumulated over all years, 
    # so we are not going to have any missing counties at all (all 2022 counties are there)
    # But we do need to drop the "excess" counties: 
    # counties that have been removed from the national county structure by the year 2022,
    # (these counties will not be plotted on any of the maps)
    variable_df = variable_df[variable_df['FIPS'].isin(shape['FIPS'])]

    # We don't need to add any missing fips codes with missing data tags b/c
    # this was taken care of in the variable_df construction with the outer joins
    
    return variable_df

def save_interim_rates(variable_df, var):

    plain_english_var = SVI_VAR_LIST.get(var, '')
    output_path = f'Data/SVI/Interim Files/{plain_english_var}_interim.csv'

    # Multiply values by 100 so that our rates are now between 0 and 100
    variable_df = variable_df.apply(
        lambda x: x.apply(lambda y: round(y * 100, 2) if y != MISSING_DATA_VALUE else y) 
        if x.name != 'FIPS' else x
    )

    # Reorder the columns for saving
    column_order = ['FIPS'] + [f'{year} {var}' for year in range(2010, 2023)]
    variable_df = variable_df[column_order]

    # Add leading zeros to the FIPS codes
    variable_df['FIPS'] = variable_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)

    # Save the datafram to a csv
    variable_df.to_csv(output_path, index=False)
    print(f'Rates saved for {var}.')

def main():
    df_2010, df_2014, df_2016, df_2018, df_2020, df_2022 = load_dataframes()
    for var, _ in SVI_VAR_LIST.items():
        variable_df = construct_var_dataframe(df_2010, df_2014, df_2016, df_2018, df_2020, df_2022, var)
        variable_df = impute_data(variable_df, var)
        variable_df = fix_rio_arriba(variable_df, var)
        variable_df = fix_connecticut(variable_df, var)
        shape = load_shapefile(SHAPE_PATH)
        variable_df = fix_fips(shape, variable_df, var)
        save_interim_rates(variable_df, var)

if __name__ == "__main__":
    main()