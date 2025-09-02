import geopandas as gpd

# Constants
SHAPEFILE_PATH = '2022 USA County Shapefile/Cartographic Boundary Files/cb_2022_us_county_20m.shp'
EXCLUDE_TERITORIES = ['03', '07', '14', '43', '52', '72']
FILTERED_SHAPEFILE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'

def load_file_and_drop(shapefile_path, exclude_territories):
    shape = gpd.read_file(shapefile_path, dtype={'STATEFP': str, 'COUNTYFP': str})
    shape = shape[~shape['STATEFP'].isin(exclude_territories)] # drop the territories we don't want
    return shape

def create_fips_codes(shape):
    # Construct the 5 digit FIPS codes
    shape['FIPS'] = shape['STATEFP'] + shape['COUNTYFP']
    shape.sort_values('FIPS', inplace=True)
    return shape

def save_filtered_shapefile(shape, filtered_shapefile_path):
    shape.to_file(filtered_shapefile_path)

def main():
    shape = load_file_and_drop(SHAPEFILE_PATH, EXCLUDE_TERITORIES)
    shape = create_fips_codes(shape)
    save_filtered_shapefile(shape, FILTERED_SHAPEFILE_PATH)

if __name__ == "__main__":
    main()