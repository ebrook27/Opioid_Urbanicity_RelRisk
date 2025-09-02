### 4/23/25, EB: Here I am trying to compute the relative risk score for each county individually. I've gotten confused
### by how best to treat the prediction of relative risk, and I had the idea to try this. If we can get an RR score
### for each county, then we're predicting a much more continuous variable, rather than a sort of discrete-regression
### with scores for each risk level. If it doesn't work, then oh well.

import pandas as pd
import os

# === Load data ===
input_file = "Data\Mortality\Final Files\Mortality_final_rates.csv"
df = pd.read_csv(input_file)

# === Reshape to long format ===
df_long = df.melt(id_vars=["FIPS"], var_name="Year_MR", value_name="mortality_rate")

# Extract the year as integer
df_long["year"] = df_long["Year_MR"].str.extract(r"(\d{4})").astype(int)

# === Compute national average per year ===
national_avg = df_long.groupby("year")["mortality_rate"].mean().rename("national_avg_rate")

# Merge back into main DataFrame
df_long = df_long.merge(national_avg, on="year", how="left")

# === Compute relative risk ===
df_long["relative_risk"] = df_long["mortality_rate"] / df_long["national_avg_rate"]

# === Clean and save output ===
output_df = df_long[["FIPS", "year", "mortality_rate", "national_avg_rate", "relative_risk"]]
output_file = "Data\Mortality\Final Files\Mortality_RR_per_county.csv"
output_df.to_csv(output_file, index=False)

print(f"✅ Relative risk calculation complete! Saved to: {os.path.abspath(output_file)}")
