### 5/13/25, EB: Talked with Andrew on 5/8 about the issues I was having with the anomaly detection, and how it kept only finding
### cold spot counties. He suggested that I try using the autoencoder model to detect anomalies instead of the Isolation Forest anomaly detection method.
### I think this will probably find the hot spots in the mortality data, and we could try to combine it with the results of the clustering to 
### develop some sort of dimension-reduced hybrid predictive model.

### 5/14/25, EB: It seems like the autoencoder model performs reconstruction well, and, for the most part, the high reconstruction errors match up with the counties that have high mortality rates.
### I've generated plots showing, for all counties each year, the reconstruction error vs mortality rate.
### My goal now is to take these counties with high reconstruction error and see if they have any discernible patterns in their SVI data. What would be great is if this would line up with the clustering results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

torch.manual_seed(42)

class MortalityAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(MortalityAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, dtype={'FIPS': str})
    mortality_cols = [col for col in df.columns if 'MR' in col]
    FIPS = df['FIPS']
    X_raw = df[mortality_cols].values
    return df, FIPS, X_raw

def load_and_prepare_data_by_year(filepath):
    df = pd.read_csv(filepath, dtype={'FIPS': str})
    mortality_cols = [col for col in df.columns if 'MR' in col]
    
    # Strip year from column names
    year_map = {col: col.split()[0] for col in mortality_cols}
    df = df.rename(columns=year_map)
    
    # Reshape: rows = years, columns = counties
    df_long = df.melt(id_vars='FIPS', var_name='Year', value_name='Mortality')
    df_wide = df_long.pivot(index='Year', columns='FIPS', values='Mortality').fillna(0)
    
    return df_wide  # Rows = years, Columns = counties


def preprocess_data(X_raw):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return torch.tensor(X_scaled, dtype=torch.float32), scaler


def train_model(model, dataloader, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    return model


def compute_reconstruction_error(model, X_tensor):
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor).numpy()
    errors = np.mean((X_tensor.numpy() - recon) ** 2, axis=1)
    return recon, errors


def plot_top_errors(FIPS, errors, top_n=20):
    top_indices = np.argsort(errors)[-top_n:][::-1]
    top_fips = FIPS.iloc[top_indices]
    top_errors = errors[top_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(top_fips, top_errors)
    plt.xticks(rotation=90)
    plt.title(f'Top {top_n} Hot Spot Counties (by Reconstruction Error)')
    plt.xlabel('FIPS Code')
    plt.ylabel('Reconstruction Error')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_top_errors_by_year(FIPS, yearly_errors, year_idx, top_n=20, year_label=None):
    errors = yearly_errors[year_idx]
    top_indices = np.argsort(errors)[-top_n:][::-1]
    top_fips = FIPS.iloc[top_indices]
    top_errors = errors[top_indices]

    year_label = year_label or f"Year {year_idx}"

    plt.figure(figsize=(12, 6))
    plt.bar(top_fips, top_errors)
    plt.xticks(rotation=90)
    plt.title(f"Top {top_n} Hot Spot Counties in {year_label}")
    plt.xlabel("FIPS Code")
    plt.ylabel("Reconstruction Error")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

########################################################
### 5/14/25, EB: I think the model is working well, but I need to make sure that the counties that are hard to recreate are actually the ones with high mortality rates.
### I will check the reconstruction errors against the original data to see if they are actually high mortality counties.
# def plot_error_vs_mortality(year_idx, errors_df, mortality_df, fips, year_label=None, highlight_top_n=10):
#     """
#     Scatter plot of reconstruction error vs mortality rate for a given year.
    
#     Parameters:
#     - year_idx: index of the year in the DataFrames (0 = 2010, etc.)
#     - errors_df: pd.DataFrame of reconstruction errors (rows = years, columns = FIPS)
#     - mortality_df: pd.DataFrame of mortality rates (same shape as errors_df)
#     - fips: list or Index of FIPS codes (column names)
#     - year_label: actual year string to use in the title (e.g., '2017')
#     - highlight_top_n: optionally label the top-N high error counties
#     """
#     year_label = year_label or str(mortality_df.index[year_idx])
    
#     # Extract the data for the given year
#     errors = errors_df.iloc[year_idx].values
#     mortality = mortality_df.iloc[year_idx].values

#     plt.figure(figsize=(10, 6))
#     plt.scatter(mortality, errors, alpha=0.6, edgecolor='k', linewidth=0.3)
#     plt.xlabel('Mortality Rate')
#     plt.ylabel('Reconstruction Error')
#     plt.title(f'{year_label}: Reconstruction Error vs Mortality Rate')
#     plt.grid(True)

#     # Optional: label top N highest error counties
#     if highlight_top_n > 0:
#         top_indices = errors.argsort()[-highlight_top_n:][::-1]
#         for i in top_indices:
#             plt.text(mortality[i], errors[i], fips[i], fontsize=8)

#     plt.tight_layout()
#     plt.show()



def plot_error_vs_mortality(year_idx, errors_df, mortality_df, fips, 
                            year_label=None, highlight_top_n=10, 
                            save=False, save_dir='County Classification/AE_Plots/Recon_Error_vs_Mortality'):
    """
    Scatter plot of reconstruction error vs mortality rate for a given year.
    
    Parameters:
    - year_idx: index of the year in the DataFrames (0 = 2010, etc.)
    - errors_df: pd.DataFrame of reconstruction errors (rows = years, columns = FIPS)
    - mortality_df: pd.DataFrame of mortality rates (same shape as errors_df)
    - fips: list or Index of FIPS codes (column names)
    - year_label: actual year string to use in the title (e.g., '2017')
    - highlight_top_n: optionally label the top-N high error counties
    - save: whether to save the figure
    - save_dir: where to save the figure (directory will be created if needed)
    """
    year_label = year_label or str(mortality_df.index[year_idx])

    # Extract the data for the given year
    errors = errors_df.iloc[year_idx].values
    mortality = mortality_df.iloc[year_idx].values

    plt.figure(figsize=(10, 6))
    plt.scatter(mortality, errors, alpha=0.6, edgecolor='k', linewidth=0.3)
    plt.xlabel('Mortality Rate')
    plt.ylabel('Reconstruction Error')
    plt.title(f'{year_label}: Reconstruction Error vs Mortality Rate')
    plt.grid(True)

    # Optional: label top N highest error counties
    if highlight_top_n > 0:
        top_indices = errors.argsort()[-highlight_top_n:][::-1]
        for i in top_indices:
            plt.text(mortality[i], errors[i], fips[i], fontsize=8)

    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"AE_Recon_Error_vs_Mort_{year_label}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot for {year_label} to {filepath}")
    else:
        plt.show()



#################
### Here I'm adding some functions to investigate the SVI profiles for the counties with high reconstruction error.
def construct_data_df_for_year(year):
    """Returns a merged DataFrame with all SVI variables, mortality, and urban-rural class for a given year."""
    assert 2010 <= year <= 2022, "Year out of range."

    DATA = ['Mortality',
            'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
            'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
            'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
            'Single-Parent Household', 'Unemployment']

    data_df = pd.read_csv(
        'Data/Mortality/Final Files/Mortality_final_rates.csv',
        header=0,
        names=['FIPS'] + [f'{y} Mortality Rates' for y in range(2010, 2023)],
        dtype={'FIPS': str}
    )
    data_df['FIPS'] = data_df['FIPS'].str.zfill(5)
    data_df = data_df[['FIPS', f'{year} Mortality Rates']]

    for variable in [v for v in DATA if v != 'Mortality']:
        path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
        var_df = pd.read_csv(
            path,
            header=0,
            names=['FIPS'] + [f'{y} {variable}' for y in range(2010, 2023)],
            dtype={'FIPS': str}
        )
        var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
        var_df = var_df[['FIPS', f'{year} {variable}']]
        data_df = pd.merge(data_df, var_df, on='FIPS', how='left')

    # Urban-rural codes
    urban_rural = pd.read_csv(
        'Data/SVI/NCHS_urban_v_rural.csv',
        usecols=['FIPS', '2023 Code'],
        dtype={'FIPS': str}
    )
    urban_rural['FIPS'] = urban_rural['FIPS'].str.zfill(5)
    urban_rural['urban_rural_class'] = urban_rural['2023 Code'].astype(int) - 1
    data_df = pd.merge(data_df, urban_rural[['FIPS', 'urban_rural_class']], on='FIPS', how='left')

    return data_df

def compare_svi_for_high_error(errors_df, year, svi_df, fips, percentile=90):
    """
    Compare SVI features of high-error vs low-error counties for a specific year.
    """
    year = str(year)
    year_errors = errors_df.loc[year]
    cutoff = np.percentile(year_errors, percentile)

    high_fips = year_errors[year_errors > cutoff].index
    low_fips = year_errors[year_errors <= cutoff].index

    high_df = svi_df[svi_df['FIPS'].isin(high_fips)]
    low_df = svi_df[svi_df['FIPS'].isin(low_fips)]

    comparison = pd.DataFrame({
        'High Error Mean': high_df.drop(columns=['FIPS', 'urban_rural_class']).mean(numeric_only=True),
        'Low Error Mean': low_df.drop(columns=['FIPS', 'urban_rural_class']).mean(numeric_only=True)
    })
    comparison['Difference'] = comparison['High Error Mean'] - comparison['Low Error Mean']
    comparison = comparison.sort_values('Difference', ascending=False)

    return comparison





def main():
    # --- Step 1: Load & preprocess data ---
    filepath = 'Data\Mortality\Final Files\Mortality_final_rates.csv'
    #df, FIPS, X_raw = load_and_prepare_data(filepath)
    #X_tensor, scaler = preprocess_data(X_raw)
    #X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    df_wide = load_and_prepare_data_by_year(filepath)
    #FIPS = df_wide.columns
    # To fix plotting issue, we need to make FIPS a pd.Series
    FIPS = pd.Series(df_wide.columns)
    X_raw = df_wide.values
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    
    # --- Step 2: Prepare DataLoader ---
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --- Step 3: Initialize and train model ---
    model = MortalityAutoencoder(input_dim=X_tensor.shape[1])
    trained_model = train_model(model, dataloader, epochs=100)

    # --- Step 4: Compute reconstruction errors ---
    recon, errors = compute_reconstruction_error(trained_model, X_tensor)
    
     # --- Step 6: Aggregate reconstruction error per county ---
    yearly_errors = (X_tensor.numpy() - recon) ** 2  # error for each year
    #per_county_errors = np.mean(yearly_errors, axis=1)  # average error across years

    # --- Step 5: Plot top counties by reconstruction error ---
    #plot_top_errors(FIPS, per_county_errors, top_n=20)

    # #############
    # # # Plotting errors for each year
    # years = df_wide.index.tolist()  # Assuming df_wide uses years as index    
    # for i, year in enumerate(years):
    #     plot_top_errors_by_year(FIPS, yearly_errors, i, top_n=20, year_label=year)


    # #############
    # ### To compare SVI features of high-error vs low-error counties
    
    # mortality_df = pd.DataFrame(X_raw, columns=FIPS, index=df_wide.index)
    # errors_df = pd.DataFrame(yearly_errors, columns=FIPS, index=df_wide.index)
    # # year = 2016
    # # svi_2016 = construct_data_df_for_year(year)
    # # comparison_df = compare_svi_for_high_error(errors_df, year, svi_2016, FIPS, percentile=90)
    # # print(comparison_df)
    
    # all_comparisons = []

    # for year in range(2010, 2023):
    #     try:
    #         svi_df = construct_data_df_for_year(year)
    #         comparison_df = compare_svi_for_high_error(errors_df, year, svi_df, FIPS, percentile=90)
    #         comparison_df['Year'] = year
    #         comparison_df['Feature'] = comparison_df.index
    #         all_comparisons.append(comparison_df.reset_index(drop=True))
    #         print(f"Processed year {year}")
    #     except Exception as e:
    #         print(f"Failed for year {year}: {e}")

    # final_comparison_df = pd.concat(all_comparisons, ignore_index=True)
    # final_comparison_df.to_csv('County Classification/autoencoder_svi_high_error_comparison_by_year.csv', index=False)
    # print("All years saved to 'County Classification/autoencoder_svi_high_error_comparison_by_year.csv'")



    #############
    ### For looking at the reconstruction error vs mortality rate
    # --- Step 6: Prepare mortality rates in same shape ---
    mortality_df = pd.DataFrame(X_raw, columns=FIPS, index=df_wide.index)
    errors_df = pd.DataFrame(yearly_errors, columns=FIPS, index=df_wide.index)

    for i, year in enumerate(errors_df.index):
        # plot_error_vs_mortality(
        #     year_idx=i,
        #     errors_df=errors_df,
        #     mortality_df=mortality_df,
        #     fips=errors_df.columns,
        #     year_label=year,
        #     highlight_top_n=5
        # )
        plot_error_vs_mortality(
            year_idx=i,
            errors_df=errors_df,
            mortality_df=mortality_df,
            fips=errors_df.columns,
            year_label=year, 
            highlight_top_n=10, 
            save=True,
            save_dir='County Classification/AE_Plots/Recon_Error_vs_Mortality')

    # #############
    # ### For saving results to plot on accuracy maps:0
    # # Save reconstructions and errors to CSV for mapping
    # recon_df = pd.DataFrame(recon, columns=FIPS)
    # recon_df.insert(0, "Year", df_wide.index)
    # recon_df.to_csv("County Classification/AE_Preds/ae_mortality_recon.csv", index=False)

    # error_df = pd.DataFrame(yearly_errors, columns=FIPS)
    # error_df.insert(0, "Year", df_wide.index)
    # error_df.to_csv("County Classification/AE_Preds/ae_mortality_errors.csv", index=False)



if __name__ == "__main__":
    main()
