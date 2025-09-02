import pandas as pd

def load_neighbors():
    neighs_path = 'Data/Neighbors/2022_neighbors_list.csv'
    neighs_names = ['FIPS', 'Neighbors']
    neighs_df = pd.read_csv(neighs_path, header=None, names=neighs_names)

    neighs_df['FIPS'] = neighs_df['FIPS'].astype(str).str.zfill(5)
    neighs_df['Neighbors'] = neighs_df['Neighbors'].apply(
        lambda x: x.split(',') if isinstance(x, str) and ',' in x else ([] if pd.isna(x) or x == '' else [x])
    )
    return neighs_df

def load_yearly_mortality(year):
    input_path = f'Data/Mortality/Interim Files/{year}_mortality_interim.csv'
    mort_names = ['FIPS', 'Deaths', 'Population', f'{year} MR']
    cols_to_keep = ['FIPS', f'{year} MR']

    mort_df = pd.read_csv(input_path, header=0, names=mort_names)
    mort_df = mort_df[cols_to_keep]
    
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[f'{year} MR'] = mort_df[f'{year} MR'].astype(float)
    return mort_df

def fill_missing_neighbors(mort_df, neighs_df, year, num_missing):
    step_count = 0
    for fips, row in mort_df.iterrows():
        if row[f'{year} MR'] == -9.0:
            neighbors = neighs_df.loc[neighs_df['FIPS'] == fips, 'Neighbors']
            neighbors = neighbors.values[0]
            available_neighbors = [neighbor for neighbor in neighbors if neighbor in mort_df.index and mort_df.loc[neighbor, f'{year} MR'] != -9]
            missing_neighbors = [neighbor for neighbor in neighbors if neighbor in mort_df.index and mort_df.loc[neighbor, f'{year} MR'] == -9]

            if len(missing_neighbors) == num_missing and len(available_neighbors) > 0:
                new_value = sum([mort_df.loc[neighbor, f'{year} MR'] for neighbor in available_neighbors]) / len(available_neighbors)
                mort_df.loc[fips, f'{year} MR'] = new_value
                step_count += 1
    return mort_df, step_count

def fill_continental_holes(mort_df, neighs_df, year):
    # Fill in continental counties with no missing neighbors
    for fips, row in mort_df.iterrows():
        if row[f'{year} MR'] == -9.0:
            neighbors = neighs_df.loc[neighs_df['FIPS'] == fips, 'Neighbors']  # list of neighbors for this county
            neighbors = neighbors.values[0]
            neighbor_rates = [mort_df.loc[neighbor, f'{year} MR'] for neighbor in neighbors if neighbor in mort_df.index and mort_df.loc[neighbor, f'{year} MR'] != -9]

            if len(neighbor_rates) == len(neighbors) and len(neighbor_rates) > 0:  # full neighbor set is available and not empty
                new_value = sum(neighbor_rates) / len(neighbor_rates)
                mort_df.at[fips, f'{year} MR'] = new_value
    return mort_df

def handle_island_counties(mort_df, year):
    island_fips = ['25019', # Nantucket County, MA
                   '15001', #Hawaii County, HI
                   '15003', # Honolulu County, HI
                   '15007', # Kauai County, HI
                   '53055'] # San Juan County, WA
    
    # Define the neighbors for each island county
    island_neighbors = {
        '25019': ['25001', '25007'],  # Barnstable and Dukes
        '15003': ['15007', '15005', '15009'],  # Kauai, Kalawao, and Maui
        '15007': ['15003'],  # Honolulu
        '15001': ['15007', '15005'],  # Kauai and Kalawao
        '53055': ['53009', '53031', '53029', '53057', '53073']  # Clallam, Jefferson, Island, Skagit, Whatcom
    }

    # Iterate through each island FIPS code
    for fips in island_fips:
        neighbors = island_neighbors[fips]
        
        # Collect mortality rates from valid neighbors
        neighbor_rates = [mort_df.loc[neighbor, f'{year} MR'] for neighbor in neighbors if neighbor in mort_df.index and mort_df.loc[neighbor, f'{year} MR'] != -9]
        
        # Calculate the new value if there are valid neighbors
        if neighbor_rates:
            new_value = sum(neighbor_rates) / len(neighbor_rates)
            mort_df.loc[fips, f'{year} MR'] = new_value
    return mort_df

def clean_rates(mort_df, neighs_df, year):
    mort_df = mort_df.set_index('FIPS')

    while True:
        count = 0

        # Fill in counties with exactly one, two, then three missing neighbors
        for num_missing in [1, 2, 3]:
            mort_df, step_count = fill_missing_neighbors(mort_df, neighs_df, year, num_missing)
            count += step_count

        # Once all categories have been properly dealt with, break the loop
        if count == 0:
            break

    # Final steps: fill in the continental holes (counties with no missing neighbors)
    mort_df = fill_continental_holes(mort_df, neighs_df, year)

    # Final steps: handle the islands
    mort_df = handle_island_counties(mort_df, year)

    # Round the mortality rates to two decimal places
    # And one CT county came out slightly negative in 2010, so we need to clip that
    mort_df[f'{year} MR'] = mort_df[f'{year} MR'].round(2).clip(lower=0)

    mort_df = mort_df.reset_index()
    return mort_df

def main():
    neighs_df = load_neighbors()
    combined_df = pd.DataFrame()

    for year in range(2010, 2023):
        mort_df = load_yearly_mortality(year)
        mort_df = clean_rates(mort_df, neighs_df, year)

        # Merge the cleaned rates into the combined DataFrame
        if combined_df.empty:
            combined_df = mort_df
        else:
            combined_df = pd.merge(combined_df, mort_df, on='FIPS', how='outer')

    # Save the combined DataFrame with all yearly mortality rates
    output_path = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
    combined_df.to_csv(output_path, index=False)
    print('Final mortality rates saved.')

if __name__ == "__main__":
    main()