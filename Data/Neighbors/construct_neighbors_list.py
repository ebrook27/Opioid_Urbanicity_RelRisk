import geopandas as gpd
import pandas as pd

year = 2020

# Load the shapefile
shapefile = gpd.read_file(f'{year} USA County Shapefile/Filtered Files/{year}_filtered_shapefile.shp')

# Initialize an empty dictionary to store neighbors
neighbors = {}

# Iterate through each county
for index, row in shapefile.iterrows():
    county_geom = row['geometry']
    county_name = row['FIPS']
    neighbors[county_name] = []
    
    # Check for neighbors
    for idx, test_county in shapefile.iterrows():
        if county_geom.touches(test_county['geometry']):
            neighbors[county_name].append(test_county['FIPS'])

# Convert to DataFrame
data = []
for county, neighbor_list in neighbors.items():
    for neighbor in neighbor_list:
        data.append({'FIPS': county, 'Neighbors': neighbor})

neighbors_df = pd.DataFrame(data)

# Group by FIPS and aggregate neighbors into a list
neighbors_df = neighbors_df.groupby('FIPS')['Neighbors'].apply(lambda x: ','.join(x)).reset_index()

# List of island counties with no queen adjacent neighbors
island_counties = ['25019', '15003', '15007', '15001', '53055']

# Manually add these counties with empty neighbor lists
for county in island_counties:
    neighbors_df = pd.concat([neighbors_df, pd.DataFrame({'FIPS': [county], 'Neighbors': ['']})])

# Sort the DataFrame by FIPS
neighbors_df = neighbors_df.sort_values(by='FIPS').reset_index(drop=True)

# Save to CSV
neighbors_df.to_csv(f'Data/Neighbors/{year}_neighbors_list.csv', index=False)
print('Neighbor file saved.')