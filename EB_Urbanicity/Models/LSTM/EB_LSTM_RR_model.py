### 4/22/25, EB: Here I am trying to implement an LSTM model to predict the mortality rate/relative risk of each county based on its SVI data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from functools import reduce
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

### This function prepares the log(Mortality Rate) dataset for the LSTM model, which I think is the wrong move.
# def prepare_yearly_prediction_data_log_mortality():
#     """
#     Prepares a long-format dataset for predicting next-year log(Mortality Rate)
#     using current-year SVI + county class as inputs.
#     """
#     svi_variables = [v for v in DATA if v != 'Mortality']
#     years = list(range(2010, 2022))  # Predict for 2011–2022 using 2010–2021 data

#     # Load county urbanicity class
#     nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
#     nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
#     nchs_df = nchs_df.set_index('FIPS')
#     nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

#     # Load mortality data
#     mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
#     mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
#     mort_df = mort_df.set_index('FIPS')

#     # Load and reshape SVI variables
#     svi_data = []
#     for var in svi_variables:
#         var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
#         var_df = pd.read_csv(var_path, dtype={'FIPS': str})
#         var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
#         long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)
#         long_df['year'] = long_df['year_var'].str.extract(r'(\d{4})').astype(int)
#         long_df = long_df[long_df['year'].between(2010, 2021)]
#         long_df = long_df.drop(columns='year_var')
#         svi_data.append(long_df)

#     # Merge all SVI variables into one long dataframe
#     svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

#     # Add next-year log(Mortality Rate)
#     for y in years:
#         mr_col = f'{y+1} MR'
#         if mr_col not in mort_df.columns:
#             continue
#         svi_merged.loc[svi_merged['year'] == y, 'log_mortality_next'] = svi_merged.loc[
#             svi_merged['year'] == y, 'FIPS'].map(mort_df[mr_col]).apply(lambda x: np.log1p(x) if pd.notnull(x) else None)

#     # Add urbanicity class
#     svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

#     # Drop rows with missing data
#     svi_merged = svi_merged.dropna()

#     return svi_merged

### This one prepares the mortality rates directly.
def prepare_yearly_prediction_data_mortality():
    """
    Prepares a long-format dataset for predicting next-year Mortality Rate
    using current-year SVI + county class as inputs.
    """
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2022))  # Predict for 2011–2022 using 2010–2021 data

    # Load county urbanicity class
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

    # Load mortality data
    mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
    mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
    mort_df = mort_df.set_index('FIPS')

    # Load and reshape SVI variables
    svi_data = []
    for var in svi_variables:
        var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
        var_df = pd.read_csv(var_path, dtype={'FIPS': str})
        var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
        long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)
        long_df['year'] = long_df['year_var'].str.extract(r'(\d{4})').astype(int)
        long_df = long_df[long_df['year'].between(2010, 2021)]
        long_df = long_df.drop(columns='year_var')
        svi_data.append(long_df)

    # Merge all SVI variables into one long dataframe
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Add next-year Mortality Rate (no log transform)
    for y in years:
        mr_col = f'{y+1} MR'
        if mr_col not in mort_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'mortality_next'] = svi_merged.loc[
            svi_merged['year'] == y, 'FIPS'].map(mort_df[mr_col])

    # Add urbanicity class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows with missing data
    svi_merged = svi_merged.dropna()

    return svi_merged


def prepare_lstm_dataset(df, svi_variables, sequence_length=3):
    # Encode urbanicity (county_class)
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    county_class_encoded = enc.fit_transform(df[['county_class']])
    df_encoded = df.copy()
    for i, col in enumerate(enc.get_feature_names_out(['county_class'])):
        df_encoded[col] = county_class_encoded[:, i]
    
    # Normalize SVI features
    scalers = {var: StandardScaler() for var in svi_variables}
    for var in svi_variables:
        df_encoded[var] = scalers[var].fit_transform(df_encoded[[var]])

    # Group by county, sort by year, and construct sequences
    sequences = []
    targets = []
    static_features = []
    grouped = df_encoded.groupby('FIPS')
    for fips, group in grouped:
        group = group.sort_values('year')
        if len(group) < sequence_length + 1:
            continue
        svi_seq = group[svi_variables].values
        mort = group['log_mortality_next'].values
        urban = group[enc.get_feature_names_out(['county_class'])].values[0]  # static across years

        for i in range(len(group) - sequence_length):
            x_seq = svi_seq[i:i+sequence_length]
            y_target = mort[i+sequence_length]
            sequences.append(x_seq)
            targets.append(y_target)
            static_features.append(urban)

    return np.array(sequences), np.array(static_features), np.array(targets)

def prepare_lstm_dataset_with_tracking(df, svi_variables, sequence_length=3):
    """
    Prepares a long-format dataset for predicting next-year Mortality Rate
    using current-year SVI + county class as inputs.
    This version tracks FIPS and year for each sequence, for later analysis.
    The sequence_length is the number of years of SVI data used to predict the next year's mortality rate.
    """

    # Encode county class (urbanicity)
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    county_class_encoded = enc.fit_transform(df[['county_class']])
    df_encoded = df.copy()
    for i, col in enumerate(enc.get_feature_names_out(['county_class'])):
        df_encoded[col] = county_class_encoded[:, i]

    # Normalize SVI variables
    scalers = {var: StandardScaler() for var in svi_variables}
    for var in svi_variables:
        df_encoded[var] = scalers[var].fit_transform(df_encoded[[var]])

    sequences = []
    static_features = []
    targets = []
    fips_list = []
    year_list = []

    grouped = df_encoded.groupby('FIPS')

    for fips, group in grouped:
        group = group.sort_values('year')
        if len(group) < sequence_length + 1:
            continue

        svi_seq = group[svi_variables].values
        mortality_seq = group['mortality_next'].values
        urban_static = group[enc.get_feature_names_out(['county_class'])].values[0]
        years = group['year'].values

        for i in range(len(group) - sequence_length):
            x_seq = svi_seq[i:i+sequence_length]
            y_target = mortality_seq[i + sequence_length]
            target_year = years[i + sequence_length]  # year of mortality being predicted

            if np.isnan(y_target):
                continue

            sequences.append(x_seq)
            static_features.append(urban_static)
            targets.append(y_target)
            fips_list.append(fips)
            year_list.append(target_year)

    return (
        np.array(sequences),
        np.array(static_features),
        np.array(targets),
        np.array(fips_list),
        np.array(year_list)
    )


def yearly_data_split(X_seq, X_static, y, fips_arr, year_arr, year=2020):
    """
    Splits the dataset into training and validation sets based on the year.
    """
    train_mask = year_arr <= year
    val_mask   = year_arr   >= year + 1

    X_train_seq = X_seq[train_mask]
    X_train_static = X_static[train_mask]
    y_train = y[train_mask]
    fips_train = fips_arr[train_mask]
    year_train = year_arr[train_mask]

    X_val_seq = X_seq[val_mask]
    X_val_static = X_static[val_mask]
    y_val = y[val_mask]
    fips_val = fips_arr[val_mask]
    year_val = year_arr[val_mask]

    data_set = (X_train_seq, X_val_seq,
                X_train_static, X_val_static,
                y_train, y_val,
                fips_train, fips_val,
                year_train, year_val)

    return data_set

class MortalityDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, static_features, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.static_features[idx], self.targets[idx]


class LSTMMortalityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, static_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            ### 4/30/25, EB: Changed final layer to ReLU activation to avoid negative mortality rates.
            nn.ReLU()
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        last_output = lstm_out[:, -1, :]  # take last timestep output
        combined = torch.cat((last_output, x_static), dim=1)
        return self.fc(combined).squeeze(1)

def train_model(model, train_loader, val_loader, n_epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x_seq, x_static, y in train_loader:
            x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x_seq, x_static)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_seq, x_static, y in val_loader:
                x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
                y_pred = model(x_seq, x_static)
                val_loss += criterion(y_pred, y).item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    return model

def get_predictions(model, val_loader):
    model.eval()
    device = next(model.parameters()).device
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_seq, x_static, y in val_loader:
            x_seq, x_static = x_seq.to(device), x_static.to(device)
            preds = model(x_seq, x_static).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    return np.array(y_true), np.array(y_pred)

def plot_prediction_scatter(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # y = x line
    plt.xlabel("True log(Mortality Rate)")
    plt.ylabel("Predicted log(Mortality Rate)")
    plt.title("Predicted vs. True log(Mortality Rate)")
    plt.grid(True)
    plt.show()

def plot_residual_hist(y_true, y_pred):
    residuals = np.abs(y_pred - y_true)
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, bins=50, kde=False)
    plt.title("Residuals abs(Predicted - True MR)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_residual_and_prediction_hist(y_true, y_pred):
    residuals = np.abs(y_pred - y_true)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Residuals
    sns.histplot(residuals, bins=50, kde=False, ax=axes[0], color='skyblue')
    axes[0].set_title("Residuals (Predicted - True Mortality Rate)")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    # Plot 2: Predicted mortality rates
    sns.histplot(y_pred, bins=50, kde=False, ax=axes[1], color='salmon')
    axes[1].set_title("Distribution of Predicted Mortality Rates")
    axes[1].set_xlabel("Predicted Mortality Rate")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_residual_and_prediction_hist_by_year(results_df, out_dir=None):
    """
    Plots a figure for each year in the results dataframe.
    Each figure has:
      - Left: Histogram of residuals (absolute error)
      - Right: Histogram of predicted mortality rates

    If out_dir is provided, saves each plot instead of showing it.
    """
    years = sorted(results_df['Year'].unique())
    
    for year in years:
        df_year = results_df[results_df['Year'] == year].copy()
        residuals = np.abs(df_year['Pred_MR'] - df_year['True_MR'])
        preds = df_year['Pred_MR']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals
        sns.histplot(residuals, bins=50, kde=False, ax=axes[0], color='skyblue')
        axes[0].set_title(f"{year} Residuals (|Pred - True| MR)")
        axes[0].set_xlabel("Absolute Error")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True)

        # Predictions
        sns.histplot(preds, bins=50, kde=False, ax=axes[1], color='salmon')
        axes[1].set_title(f"{year} Predicted Mortality Rate Distribution")
        axes[1].set_xlabel("Predicted Mortality Rate")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True)

        plt.tight_layout()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/residual_and_prediction_hist_{year}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    

def main():
    print('Running LSTM model for mortality prediction...')
    # Prepare the dataset
    df = prepare_yearly_prediction_data_mortality()
    X_seq, X_static, y, fips_arr, year_arr = prepare_lstm_dataset_with_tracking(df, svi_variables=DATA[1:])

    # Want to do temporal predictions, so split the data by year
    data_set = yearly_data_split(X_seq, X_static, y, fips_arr, year_arr, year=2016)
    X_train_seq, X_val_seq, X_train_static, X_val_static, y_train, y_val, fips_train, fips_val, year_train, year_val = data_set

    # Sanity check on what years I'm using to train and validate
    print("Training years:", np.unique(year_train))
    print("Validation years:", np.unique(year_val))
    
    # Dataloaders
    train_ds = MortalityDataset(X_train_seq, X_train_static, y_train)
    val_ds = MortalityDataset(X_val_seq, X_val_static, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # Model
    model = LSTMMortalityPredictor(input_size=X_seq.shape[2], hidden_size=64, static_size=X_static.shape[1])
    trained_model = train_model(model, train_loader, val_loader, n_epochs=100)
    print('Model training complete.')

    # Get preditions
    y_true, y_pred = get_predictions(trained_model, val_loader)
    # plot_residual_and_prediction_hist(y_true, y_pred)

    results_df = pd.DataFrame({
    'FIPS': fips_val,
    'Year': year_val,
    'True_MR': y_true,
    'Pred_MR': y_pred
    })
    
    plot_residual_and_prediction_hist_by_year(results_df, out_dir='County Classification/LSTM_Plots')

    # print('Saving predictions...')
    # ### Save predictions to CSV
    # OUT_CSV_DIR = 'County Classification/LSTM_Preds'
    # os.makedirs(OUT_CSV_DIR, exist_ok=True)

    # for year in sorted(results_df['Year'].unique()):
    #     yearly_df = results_df[results_df['Year'] == year].copy()
    #     yearly_df[f'{year}_True_MR'] = yearly_df['True_MR']  # Rename to match script
    #     yearly_df[f'{year}_Pred_MR'] = yearly_df['Pred_MR']
    #     yearly_df[['FIPS', f'{year}_True_MR', f'{year}_Pred_MR']].to_csv(
    #         f'{OUT_CSV_DIR}/{year}_MR_predictions.csv', index=False
    #     )
    # print('Predictions saved to CSV files.')

if __name__ == "__main__":
    main()






########################################################################################################################################################################################################################################

######################
### 4/30/25, EB: Switched to if __name__ == "__main__" to allow for easier testing of the LSTM model.
### Took the following code from here and put it into the main function.
######################


# ### 4/23/25, EB: Testing the LSTM model on the persistently high risk counties.
# # persistent_df = pd.read_csv('Data\Mortality\Final Files\Mortality_top10_percent_counties_10yrs_lognormal.csv', dtype={'FIPS': str})
# # persistent_fips_set = set(persistent_df['FIPS'].str.zfill(5))

# # Get processed data
# # df = prepare_yearly_prediction_data_log_mortality()
# # X_seq, X_static, y = prepare_lstm_dataset(df, svi_variables=DATA[1:])


# df = prepare_yearly_prediction_data_mortality()
# X_seq, X_static, y, fips_arr, year_arr = prepare_lstm_dataset_with_tracking(df, svi_variables=DATA[1:])

# # ## Persistent counties
# # # Mask rows by whether the FIPS code is persistent
# # persistent_mask = np.array([fips in persistent_fips_set for fips in fips_arr])
# # non_persistent_mask = ~persistent_mask

# # # Separate data
# # X_seq_persist = X_seq[persistent_mask]
# # X_static_persist = X_static[persistent_mask]
# # y_persist = y[persistent_mask]
# # fips_persist = fips_arr[persistent_mask]
# # year_persist = year_arr[persistent_mask]

# # X_seq_other = X_seq[non_persistent_mask]
# # X_static_other = X_static[non_persistent_mask]
# # y_other = y[non_persistent_mask]
# # fips_other = fips_arr[non_persistent_mask]
# # year_other = year_arr[non_persistent_mask]


# # Train/val split
# # X_train_seq, X_val_seq, X_train_static, X_val_static, y_train, y_val = train_test_split(
# #     X_seq, X_static, y, test_size=0.2, random_state=42
# # )

# # (
# #     X_train_seq, X_val_seq,
# #     X_train_static, X_val_static,
# #     y_train, y_val,
# #     fips_train, fips_val,
# #     year_train, year_val
# # ) = train_test_split(
# #     X_seq, X_static, y, fips_arr, year_arr,
# #     test_size=0.2, random_state=42
# # )


# # (
# #     X_train_seq, X_val_seq,
# #     X_train_static, X_val_static,
# #     y_train, y_val,
# #     fips_train, fips_val,
# #     year_train, year_val
# # ) = train_test_split(
# #     X_seq_other, X_static_other, y_other, fips_other, year_other,
# #     test_size=0.2, random_state=42
# # )

# # X_val_seq = np.concatenate([X_val_seq, X_seq_persist])
# # X_val_static = np.concatenate([X_val_static, X_static_persist])
# # y_val = np.concatenate([y_val, y_persist])
# # fips_val = np.concatenate([fips_val, fips_persist])
# # year_val = np.concatenate([year_val, year_persist])

# data_set = yearly_data_split(X_seq, X_static, y, fips_arr, year_arr, year=2020)
# X_train_seq, X_val_seq, X_train_static, X_val_static, y_train, y_val, fips_train, fips_val, year_train, year_val = data_set

# # Dataloaders
# train_ds = MortalityDataset(X_train_seq, X_train_static, y_train)
# val_ds = MortalityDataset(X_val_seq, X_val_static, y_val)
# train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=64)

# # Model
# model = LSTMMortalityPredictor(input_size=X_seq.shape[2], hidden_size=64, static_size=X_static.shape[1])
# trained_model = train_model(model, train_loader, val_loader, n_epochs=75)


# y_true, y_pred = get_predictions(trained_model, val_loader)

# # results_df = pd.DataFrame({
# #     'FIPS': fips_val,
# #     'Year': year_val,
# #     'True_logMR': y_true,
# #     'Pred_logMR': y_pred,
# #     'True_MR': np.expm1(y_true),
# #     'Pred_MR': np.expm1(y_pred)
# # })


# results_df = pd.DataFrame({
#     'FIPS': fips_val,
#     'Year': year_val,
#     'True_MR': y_true,
#     'Pred_MR': y_pred
# })

# ### Save predictions to CSV
# OUT_CSV_DIR = 'County Classification/LSTM_Preds'
# os.makedirs(OUT_CSV_DIR, exist_ok=True)

# for year in sorted(results_df['Year'].unique()):
#     yearly_df = results_df[results_df['Year'] == year].copy()
#     yearly_df[f'{year}_True_MR'] = yearly_df['True_MR']  # Rename to match script
#     yearly_df[f'{year}_Pred_MR'] = yearly_df['Pred_MR']
#     yearly_df[['FIPS', f'{year}_True_MR', f'{year}_Pred_RR']].to_csv(
#         f'{OUT_CSV_DIR}/{year}_MR_predictions.csv', index=False
#     )


# # plot_prediction_scatter(y_true, y_pred)
# # plot_residual_hist(y_true, y_pred)
