import pandas as pd
import geopandas as gpd
import logging

RAW_MORTALITY_NAMES = ['FIPS', 'Deaths', 'Population', 'Crude Rate']
COLUMNS_TO_KEEP = ['FIPS', 'Deaths', 'Population']
MISSING_VALUE = -9

log_file = 'Log Files/missing_data.log'
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def load_yearly_mortality(year):
    input_path = f'Data/Mortality/Interim Files/{year}_mortality_interim.csv'
    mort_names = ['FIPS', 'Deaths', 'Population', f'{year} MR']
    cols_to_keep = ['FIPS', f'{year} MR']

    mort_df = pd.read_csv(input_path, header=0, names=mort_names)
    mort_df = mort_df[cols_to_keep]
    
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[f'{year} MR'] = mort_df[f'{year} MR'].astype(float)
    return mort_df

def count_missing_values(mort_df, year):
    missing_count = (mort_df[f'{year} MR'] == MISSING_VALUE).sum()
    logging.info(f'The number of missing counties in {year} is: {missing_count}')

def main():
    for year in range(2010,2023):
        mort_df = load_yearly_mortality(year)
        count_missing_values(mort_df, year)

if __name__ == "__main__":
    main()