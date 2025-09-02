import pandas as pd
import logging

log_file = 'Log Files/missing_data.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def load_interim_data(year):
    mort_path = f'Data/Mortality/Interim Files/{year}_mortality_interim.csv'
    mort_names = ['FIPS', f'{year} Deaths', f'{year} Pop', f'{year} MR']
    mort_df = pd.read_csv(mort_path, header=0, names=mort_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[f'{year} MR'] = mort_df[f'{year} MR'].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def count_missing_values(mort_df, year):
    zero_count = (mort_df[f'{year} MR'] == -9).sum()
    logging.info(f'Year {year}: {zero_count} counties are missing data.')

def main():
    for year in range(2010, 2023):
        mort_df = load_interim_data(year)
        count_missing_values(mort_df, year)

if __name__ == "__main__":
    main()