import pandas as pd
import logging

log_file = 'Log Files/missing_data.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

RAW_MORTALITY_NAMES = ['FIPS', 'Deaths', 'Population', 'Crude Rate']
COLUMNS_TO_KEEP = ['FIPS', 'Deaths', 'Population']
MISSING_VALUE = -9

def clean_rates(year):
    input_path = f'Data/Mortality/Raw Files/{year}_cdc_wonder_raw_mortality.csv'
    try:
        mort_df = pd.read_csv(input_path, header=0, names=RAW_MORTALITY_NAMES)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}")
        return None

    # Format the columns
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df['Deaths'] = mort_df['Deaths'].astype(str)  # Keep as strings initially
    mort_df['Population'] = mort_df['Population'].astype(str)

    # Keep only the required columns
    mort_df = mort_df[COLUMNS_TO_KEEP]

    # Replace "Missing/Not Available" values with -9
    mort_df['Deaths'] = mort_df['Deaths'].replace(['Missing', 'Not Available'], MISSING_VALUE)
    mort_df['Population'] = mort_df['Population'].replace(['Missing', 'Not Available'], MISSING_VALUE)

    # Convert "Deaths" and "Population" to numeric, setting errors='coerce' to handle invalid values
    mort_df['Deaths'] = pd.to_numeric(mort_df['Deaths'], errors='coerce')
    mort_df['Population'] = pd.to_numeric(mort_df['Population'], errors='coerce')

    return mort_df

def count_missing_values(mort_df, year):
    if mort_df is None:
        logging.warning(f"No data available for year {year}.")
        return

    missing_deaths = (mort_df['Deaths'] == MISSING_VALUE).sum()
    missing_population = (mort_df['Population'] == MISSING_VALUE).sum()

    logging.info(f"Year {year}: Missing Deaths = {missing_deaths}, Missing Population = {missing_population}")

def main():
    for year in range(2010, 2023):
        mort_df = clean_rates(year)
        count_missing_values(mort_df, year)

if __name__ == "__main__":
    main()
