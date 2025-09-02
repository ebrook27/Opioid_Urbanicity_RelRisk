### 8/29/25, EB: To try to predict relative risk, I think I will need a file that contains the population estimates
### for all of the counties in our study, for all of the years in our study. This script aims to create that file.
### I found two census files that contain population estimates for the years 2010-2020 adn 2020-2024. 
### I also found a page that contains the county FIPS codes for all US counties. This script cleans and scrapes the data
### from the two census files, and then merges everything on the FIPS column to create a data file similar to the 
### SVI and mortality data files, but containing population in each yearly column.
### https://www.census.gov/data/tables/time-series/demo/popest/intercensal-2010-2020-counties.html
### https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html
### https://agacis.rcc-acis.org/help/fipsList.txt

import pandas as pd

# === Step 1: Load FIPS lookup text file ===
fips_df = pd.read_csv("Data\Population\county_fips_codes.txt", sep="\t|,", engine="python", header=None, names=["State", "FIPS", "County"])
fips_df["CountyState_clean"] = (fips_df["County"] + ", " + fips_df["State"]).str.lower().str.strip()

# === Step 2: Load Census Excel files ===
pop_2010_2020 = pd.read_excel("Data\Population\co-est2020int-pop.xlsx", skiprows=3)
pop_2020_2024 = pd.read_excel("Data\Population\co-est2024-pop.xlsx", skiprows=3)

# === Step 3: Rename 2020 column in older file (it's unlabeled) ===
pop_2010_2020 = pop_2010_2020.rename(columns={"Unnamed: 12": 2020})

# === Step 4: Define cleaning function ===
def clean_population_df(df, years):
    df = df[df['Unnamed: 0'].str.startswith('.')]  # Keep county rows only
    df['CountyState_clean'] = df['Unnamed: 0'].str.lstrip('.').str.lower().str.strip()
    df_cleaned = df[['CountyState_clean'] + years]
    return df_cleaned

# === Step 5: Clean and extract year ranges ===
years_2010_2020 = list(range(2010, 2021))
years_2021_2022 = list(range(2021, 2023))

pop_10_20_clean = clean_population_df(pop_2010_2020, years_2010_2020)
pop_21_22_clean = clean_population_df(pop_2020_2024, years_2021_2022)

# === Step 6: Merge both datasets ===
pop_all_years = pd.merge(
    pop_10_20_clean,
    pop_21_22_clean,
    on="CountyState_clean",
    how="outer"
)

# === Step 7: Merge in FIPS codes ===
pop_all_years = pd.merge(
    fips_df[['FIPS', 'CountyState_clean']],
    pop_all_years,
    on="CountyState_clean",
    how="right"
)

# === Step 8: Reorder and rename columns ===
final_years = [str(y) for y in years_2010_2020 + years_2021_2022]
pop_all_years.columns = ['FIPS', 'CountyState_clean'] + final_years
pop_all_years = pop_all_years[['FIPS'] + final_years]

# === Step 8.5: Rename year columns to "20XX POP"
pop_all_years.rename(
    columns={year: f"{year} POP" for year in final_years},
    inplace=True
)

# === Step 9: Save to CSV ===
pop_all_years.to_csv("Data\Population\county_population_2010_2022.csv", index=False)

print("✅ Saved as 'county_population_2010_2022.csv'")
