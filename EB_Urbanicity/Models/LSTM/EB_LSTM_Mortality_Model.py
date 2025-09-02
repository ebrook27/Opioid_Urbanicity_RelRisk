### 6/16/25, EB: SHowed Adam and Andrew the MLP model, along with the RF feature weighting pipeline. They had some critiques
### and ideas, but for the most part like where we're heading. Adam did suggest that we try a LSTM model, which I think is a good idea.
### I tried it before when we were trying to just predict relative risk, but that was such a difficult target, nothing worked great.
### Here I'm trying to predict the mortality rate, which is a much easier target. I will try to reuse some of the old code.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import reduce
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import random

def set_random_seeds(seed=713):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

### This one prepares the mortality rates directly.
def prepare_yearly_prediction_data_mortality():
    """
    Prepares a long-format dataset for predicting next-year Mortality Rate
    using current-year SVI + county class as inputs.
    """
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2023))

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
        long_df = long_df[long_df['year'].between(2010, 2022)]
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

def prepare_lstm_dataset_with_tracking(df, svi_variables, sequence_length=3):
    """
    Prepares LSTM-ready sequences for mortality prediction.
    Each X is a (sequence_length, n_features) array of SVI data,
    with an optional static feature vector (urbanicity one-hot).
    Returns sequences, static features, targets, FIPS, and years.
    """

    # Encode county_class (urbanicity) as one-hot
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    county_class_encoded = enc.fit_transform(df[['county_class']])
    df_encoded = df.copy()
    for i, col in enumerate(enc.get_feature_names_out(['county_class'])):
        df_encoded[col] = county_class_encoded[:, i]

    # Normalize SVI variables individually
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

        # Verify urbanicity is consistent
        urban_vals = group[enc.get_feature_names_out(['county_class'])].drop_duplicates()
        if len(urban_vals) != 1:
            print(f"⚠️ Urbanicity mismatch for FIPS {fips}, skipping.")
            continue
        urban_static = urban_vals.values[0]

        svi_seq = group[svi_variables].values
        mortality_seq = group['mortality_next'].values
        years = group['year'].values

        for i in range(len(group) - sequence_length):
            x_seq = svi_seq[i:i+sequence_length]
            y_target = mortality_seq[i + sequence_length]
            target_year = years[i + sequence_length]

            if np.isnan(y_target):
                continue

            sequences.append(x_seq)
            static_features.append(urban_static)
            targets.append(y_target)
            fips_list.append(fips)
            year_list.append(target_year)

    print(f"✅ Built {len(sequences)} valid sequences.")
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


# class LSTMMortalityPredictor(nn.Module):
#     def __init__(self, input_size, hidden_size, static_size, num_layers=1, dropout=0.2):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size + static_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             ### 4/30/25, EB: Changed final layer to ReLU activation to avoid negative mortality rates.
#             nn.ReLU()
#         )

#     def forward(self, x_seq, x_static):
#         lstm_out, _ = self.lstm(x_seq)
#         last_output = lstm_out[:, -1, :]  # take last timestep output
#         combined = torch.cat((last_output, x_static), dim=1)
#         return self.fc(combined).squeeze(1)

class LSTMMortalityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, static_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # or use nn.Softplus() if smoother gradient desired
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        last_output = lstm_out[:, -1, :]
        combined = torch.cat((last_output, x_static), dim=1)
        return self.fc(combined).squeeze(1)


def train_model(model, train_loader, val_loader, n_epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x_seq, x_static, y in train_loader:
            x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x_seq, x_static)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            # Print every 10 epochs or last epoch
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    if best_model_state:
        model.load_state_dict(best_model_state)

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


##################################################
### 7/1/25, EB: Looking to do more analysis on the LSTM model, so I'm going to define some new functions here.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

def evaluate_predictions(results_df):
    y_true = results_df['True_MR'].values
    y_pred = results_df['Pred_MR'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'SpearmanR': spearman_corr
    }

### 7/2/25, EB: I'm now trying to run the LSTM model including the urbanicity class as a static feature, versus just the SVI data.
### My hope is that the urbanicity will help the model learn better, which would add more evidence to its importance.

def yearly_data_split_v2(X_seq, X_static, y, fips_arr, year_arr, year=2020, include_urbanicity=True):
    """
    Splits the dataset into training and validation sets based on the year.
    Optionally includes static urbanicity features.
    """
    train_mask = year_arr <= year
    val_mask   = year_arr >= year + 1

    X_train_seq = X_seq[train_mask]
    X_val_seq   = X_seq[val_mask]
    y_train     = y[train_mask]
    y_val       = y[val_mask]
    fips_train  = fips_arr[train_mask]
    fips_val    = fips_arr[val_mask]
    year_train  = year_arr[train_mask]
    year_val    = year_arr[val_mask]

    if include_urbanicity:
        X_train_static = X_static[train_mask]
        X_val_static   = X_static[val_mask]

        return (X_train_seq, X_val_seq,
                X_train_static, X_val_static,
                y_train, y_val,
                fips_train, fips_val,
                year_train, year_val)
    else:
        # Return None for static features when not used
        return (X_train_seq, X_val_seq,
                None, None,
                y_train, y_val,
                fips_train, fips_val,
                year_train, year_val)


def prepare_lstm_dataset_with_tracking_v2(df, svi_variables, sequence_length=3, include_urbanicity=True):
    """
    Prepares LSTM-ready sequences for mortality prediction.
    Set include_urbanicity=False to exclude urbanicity one-hot.
    """
    df_encoded = df.copy()

    if include_urbanicity:
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        county_class_encoded = enc.fit_transform(df[['county_class']])
        for i, col in enumerate(enc.get_feature_names_out(['county_class'])):
            df_encoded[col] = county_class_encoded[:, i]
        static_cols = enc.get_feature_names_out(['county_class'])
    else:
        static_cols = []

    # Normalize SVI variables
    scalers = {var: StandardScaler() for var in svi_variables}
    for var in svi_variables:
        df_encoded[var] = scalers[var].fit_transform(df_encoded[[var]])

    sequences, static_features, targets, fips_list, year_list = [], [], [], [], []
    grouped = df_encoded.groupby('FIPS')

    for fips, group in grouped:
        group = group.sort_values('year')
        if len(group) < sequence_length + 1:
            continue

        if include_urbanicity:
            urban_vals = group[list(static_cols)].drop_duplicates()
            if len(urban_vals) != 1:
                print(f"⚠️ Urbanicity mismatch for FIPS {fips}, skipping.")
                continue
            urban_static = urban_vals.values[0]

        svi_seq = group[svi_variables].values
        mortality_seq = group['mortality_next'].values
        years = group['year'].values

        for i in range(len(group) - sequence_length):
            x_seq = svi_seq[i:i+sequence_length]
            y_target = mortality_seq[i + sequence_length]
            target_year = years[i + sequence_length]

            if np.isnan(y_target):
                continue

            sequences.append(x_seq)
            static_features.append(urban_static if include_urbanicity else np.zeros(1))
            targets.append(y_target)
            fips_list.append(fips)
            year_list.append(target_year)

    print(f"✅ Built {len(sequences)} valid sequences.")
    return (
        np.array(sequences),
        np.array(static_features),
        np.array(targets),
        np.array(fips_list),
        np.array(year_list)
    )

class MortalityDataset_v2(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, static_features=None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
        # Handle optional static features
        if static_features is not None:
            self.static_features = torch.tensor(static_features, dtype=torch.float32)
        else:
            self.static_features = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.static_features is not None:
            return self.sequences[idx], self.static_features[idx], self.targets[idx]
        else:
            return self.sequences[idx], self.targets[idx]


# class LSTMMortalityPredictor_v2(nn.Module):
#     def __init__(self, input_size, hidden_size, static_size=0, num_layers=1, dropout=0.0):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
#         if static_size > 0:
#             self.fc = nn.Sequential(
#                 nn.Linear(hidden_size + static_size, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 1),
#                 nn.ReLU()
#             )
#             self.use_static = True
#         else:
#             self.fc = nn.Sequential(
#                 nn.Linear(hidden_size, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 1),
#                 nn.ReLU()
#             )
#             self.use_static = False

#     def forward(self, x_seq, x_static):
#         lstm_out, _ = self.lstm(x_seq)
#         last_output = lstm_out[:, -1, :]
#         if self.use_static:
#             combined = torch.cat((last_output, x_static), dim=1)
#         else:
#             combined = last_output
#         return self.fc(combined).squeeze(1)

class LSTMMortalityPredictor_v2(nn.Module):
    def __init__(self, input_size, hidden_size, static_size=0, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.use_static = static_size > 0
        if self.use_static:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size + static_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.ReLU()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.ReLU()
            )

    def forward(self, x_seq, x_static=None):
        lstm_out, _ = self.lstm(x_seq)
        last_output = lstm_out[:, -1, :]
        if self.use_static:
            if x_static is None:
                raise ValueError("Model was initialized with static input, but x_static was not provided.")
            combined = torch.cat((last_output, x_static), dim=1)
        else:
            combined = last_output
        return self.fc(combined).squeeze(1)


def train_model_v2(model, train_loader, val_loader, n_epochs=30, lr=1e-3, use_urbanicity=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if use_urbanicity:
                x_seq, x_static, y = batch
                x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
                y_pred = model(x_seq, x_static)
            else:
                x_seq, y = batch
                x_seq, y = x_seq.to(device), y.to(device)
                y_pred = model(x_seq)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if use_urbanicity:
                    x_seq, x_static, y = batch
                    x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
                    y_pred = model(x_seq, x_static)
                else:
                    x_seq, y = batch
                    x_seq, y = x_seq.to(device), y.to(device)
                    y_pred = model(x_seq)

                val_loss += criterion(y_pred, y).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

def get_predictions_v2(model, val_loader, use_urbanicity=True):
    model.eval()
    device = next(model.parameters()).device
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in val_loader:
            if use_urbanicity:
                x_seq, x_static, y = batch
                x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
                preds = model(x_seq, x_static)
            else:
                x_seq, y = batch
                x_seq, y = x_seq.to(device), y.to(device)
                preds = model(x_seq)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    return np.array(y_true), np.array(y_pred)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_comparison_residual_and_prediction_hist_by_year(results_df_1, results_df_2, label_1='With Urbanicity', label_2='Without Urbanicity', out_dir=None, sequence_length=None):
    """
    Plots overlaid histograms of residuals and predicted mortality rates for two models.
    For each year:
      - Left: Histogram of residuals (absolute error) for both models
      - Right: Histogram of predicted mortality rates for both models

    Parameters:
        results_df_1: DataFrame with columns ['FIPS', 'Year', 'True_MR', 'Pred_MR'] for model 1
        results_df_2: DataFrame for model 2
        label_1: Label for model 1 (e.g., "With Urbanicity")
        label_2: Label for model 2 (e.g., "Without Urbanicity")
        out_dir: If specified, saves each plot instead of displaying
    """
    years = sorted(set(results_df_1['Year']).intersection(results_df_2['Year']))

    for year in years:
        df1 = results_df_1[results_df_1['Year'] == year].copy()
        df2 = results_df_2[results_df_2['Year'] == year].copy()

        # Make sure data are aligned
        df1 = df1.set_index('FIPS')
        df2 = df2.set_index('FIPS')
        common_fips = df1.index.intersection(df2.index)

        df1 = df1.loc[common_fips]
        df2 = df2.loc[common_fips]

        # Compute residuals
        residuals_1 = np.abs(df1['Pred_MR'] - df1['True_MR'])
        residuals_2 = np.abs(df2['Pred_MR'] - df2['True_MR'])

        # Predictions
        preds_1 = df1['Pred_MR']
        preds_2 = df2['Pred_MR']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals
        sns.histplot(residuals_1, bins=50, kde=False, ax=axes[0], label=label_1, color='skyblue', stat='density', alpha=0.6)
        sns.histplot(residuals_2, bins=50, kde=False, ax=axes[0], label=label_2, color='orange', stat='density', alpha=0.6)
        axes[0].set_title(f"{year} Residuals (|Pred - True| MR)")
        axes[0].set_xlabel("Absolute Error")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True)

        # Predictions
        sns.histplot(preds_1, bins=50, kde=False, ax=axes[1], label=label_1, color='salmon', stat='density', alpha=0.6)
        sns.histplot(preds_2, bins=50, kde=False, ax=axes[1], label=label_2, color='green', stat='density', alpha=0.6)
        axes[1].set_title(f"{year} Predicted Mortality Rate Distribution")
        axes[1].set_xlabel("Predicted Mortality Rate")
        axes[1].set_ylabel("Density")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/compare_LSTMs_hist_{year}_using_{sequence_length}_years.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


### 7/7/25, EB: So using the previous function and doing a t-test on the residuals from both models, it seems like including the urbanicity
### does help improve the prediction somewhat, but its significance goes down as sequence length increases.
### Here I am making a function that will create boxplots visualizing the improvement in performance by category. Hopefully we see something
### interesting here, I'm really not sure what to expect.

def plot_error_boxplots_by_urbanicity(results_with_urb, results_without_urb, urbanicity_path):
    """
    Generates boxplots comparing absolute errors by urbanicity class
    for LSTM models with and without urbanicity information.

    Parameters:
    - results_with_urb: pd.DataFrame with columns ['FIPS', 'Year', 'True_MR', 'Pred_MR']
    - results_without_urb: same structure
    - urbanicity_path: path to CSV containing ['FIPS', '2023 Code'] (urbanicity class)
    """
    # Load urbanicity class info
    urb = pd.read_csv(urbanicity_path, dtype={'FIPS': str})
    urb['FIPS'] = urb['FIPS'].str.zfill(5)
    urb = urb.rename(columns={'2023 Code': 'county_class'})

    # Add residuals to both result sets
    for df, label in zip([results_with_urb, results_without_urb], ['With Urbanicity', 'Without Urbanicity']):
        df['abs_error'] = (df['Pred_MR'] - df['True_MR']).abs()
        df['model'] = label

    # Combine the two
    combined = pd.concat([results_with_urb, results_without_urb])
    
    # Merge with urbanicity
    combined = combined.merge(urb[['FIPS', 'county_class']], on='FIPS', how='left')
    combined = combined.dropna(subset=['county_class'])

    # Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined, x='county_class', y='abs_error', hue='model')
    plt.title("Absolute Error by Urbanicity Class")
    plt.xlabel("Urbanicity Class")
    plt.ylabel("Absolute Error (|Predicted - True| Mortality Rate)")
    plt.legend(title="Model Type")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


### 7/8/25, EB: Trying to use SHAP to analyze the LSTM model with urbanicity. This is a bit tricky since LSTMs are not
### natively supported by SHAP, and we have the static features as well. Worth a shot.

import shap

def compute_shap_values_for_lstm_with_static(model, X_seq, X_static, sequence_length, feature_names_seq, feature_names_static, num_samples=50, background_size=100):
    """
    Computes SHAP values for an LSTM model with static inputs (e.g. urbanicity).
    
    Parameters:
        model: Trained PyTorch model with LSTM + static input.
        X_seq: Validation time-series data of shape (N, seq_len, num_features)
        X_static: Validation static data (e.g., one-hot urbanicity), shape (N, num_static_features)
        sequence_length: Length of the time series sequences
        feature_names_seq: List of names of the time-series features (e.g., SVI variable names)
        feature_names_static: List of names of static features (e.g., ['Urbanicity_1', ..., 'Urbanicity_k'])
        num_samples: Number of samples to explain
        background_size: Size of background dataset for SHAP

    Returns:
        shap_values: The computed SHAP values
        feature_names: Names of flattened features for interpretability
    """
    device = next(model.parameters()).device
    model.eval()

    # Flatten sequence input
    X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
    X_combined = np.hstack([X_seq_flat, X_static])

    # Feature names
    seq_names = [f"{var}_t-{sequence_length - 1 - t}" for t in range(sequence_length) for var in feature_names_seq]
    all_feature_names = seq_names + feature_names_static

    # Background data
    background = X_combined[np.random.choice(len(X_combined), size=background_size, replace=False)]

    # Wrapper
    class LSTMWithStaticSHAPWrapper:
        def __init__(self, model, seq_len, seq_feats, static_feats, device='cpu'):
            self.model = model.to(device)
            self.device = device
            self.seq_len = seq_len
            self.seq_feats = seq_feats
            self.static_feats = static_feats

        def __call__(self, X_combined):
            seq_part = X_combined[:, :self.seq_len * self.seq_feats]
            static_part = X_combined[:, self.seq_len * self.seq_feats:]
            x_seq = torch.tensor(seq_part, dtype=torch.float32).reshape(-1, self.seq_len, self.seq_feats).to(self.device)
            x_static = torch.tensor(static_part, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                preds = self.model(x_seq, x_static)
            return preds.cpu().numpy()

    wrapper = LSTMWithStaticSHAPWrapper(
        model=model,
        seq_len=sequence_length,
        seq_feats=X_seq.shape[2],
        static_feats=X_static.shape[1],
        device=device
    )

    # Run SHAP
    explainer = shap.KernelExplainer(wrapper, background)
    X_sample = X_combined[:num_samples]
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    shap.summary_plot(shap_values, features=X_sample, feature_names=all_feature_names)

    # Return raw SHAP values and names for further analysis
    return shap_values, all_feature_names

def summarize_shap_importance(shap_values, feature_names, top_n=20, plot=True):
    """
    Summarizes and ranks feature importance from SHAP values.

    Parameters:
        shap_values: SHAP values array (shape: [n_samples, n_features])
        feature_names: List of feature names corresponding to SHAP columns
        top_n: Number of top features to show
        plot: Whether to plot a horizontal bar chart

    Returns:
        importance_df: Pandas DataFrame with feature importances sorted
    """
    mean_abs_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_importance
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)

    if plot:
        top_features = importance_df.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'][::-1], top_features['Mean_Abs_SHAP'][::-1], color='slateblue')
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title("Top SHAP Feature Importances")
        plt.tight_layout()
        plt.grid(True, axis='x')
        plt.show()

    return importance_df

### 7/9/25, EB: I want to now investigate the performance of the LSTM model by urbanicity category.
### Before I can create error histograms, I need to attach the urbanicity class to the results dataframe.

def attach_urbanicity(results_df, urbanicity_path='Data/SVI/NCHS_urban_v_rural.csv'):
    """
    Adds county urbanicity class to results_df via FIPS match.

    Parameters:
        results_df: DataFrame with model predictions
        urbanicity_path: Path to the CSV with FIPS and urbanicity code

    Returns:
        results_with_urban: DataFrame with added 'county_class' column
    """
    urban_df = pd.read_csv(urbanicity_path, dtype={'FIPS': str})
    urban_df['FIPS'] = urban_df['FIPS'].str.zfill(5)
    results_df['FIPS'] = results_df['FIPS'].astype(str).str.zfill(5)

    merged = results_df.merge(urban_df[['FIPS', '2023 Code']], on='FIPS', how='left')
    merged = merged.rename(columns={'2023 Code': 'county_class'})
    return merged

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_error_histograms_by_urbanicity(results_with_urban, bins=40, out_dir=None):
    """
    Creates a 2x3 grid of absolute error histograms per urbanicity category for each year.

    Parameters:
        results_with_urban: DataFrame with 'Abs_Error', 'county_class', 'Year'
        bins: Number of bins in the histograms
        out_dir: If provided, saves each figure to disk instead of showing
    """
    if 'Abs_Error' not in results_with_urban.columns:
        results_with_urban['Abs_Error'] = np.abs(results_with_urban['Pred_MR'] - results_with_urban['True_MR'])

    urban_classes = sorted(results_with_urban['county_class'].dropna().unique())
    years = sorted(results_with_urban['Year'].unique())

    n_rows, n_cols = 2, 3

    for year in years:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
        fig.suptitle(f"Absolute Error by Urbanicity Class — Year {year}", fontsize=16)
        df_year = results_with_urban[results_with_urban['Year'] == year]

        for idx, urban_class in enumerate(urban_classes):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            subset = df_year[df_year['county_class'] == urban_class]

            sns.histplot(subset['Abs_Error'], bins=bins, ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f"Urbanicity Class {urban_class}")
            ax.set_xlabel("Absolute Error")
            ax.set_ylabel("Count")
            ax.grid(True)

        # Hide any unused subplots
        total_subplots = n_rows * n_cols
        for idx in range(len(urban_classes), total_subplots):
            row, col = divmod(idx, n_cols)
            fig.delaxes(axes[row][col])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if out_dir:
            import os
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/error_histograms_{year}.png", dpi=300)
            plt.close()
        else:
            plt.show()



#################################################################################################################
### 7/6/25,EB: Combining main functions to be clever when comparing models with and without urbanicity.
def lstm_with_urb(sequence_length=2):
    set_random_seeds()
    #sequence_length = 2  # Number of previous years to use for prediction
    
    print('Running LSTM model for mortality prediction...')
    # Prepare the dataset
    df = prepare_yearly_prediction_data_mortality()
    X_seq, X_static, y, fips_arr, year_arr = prepare_lstm_dataset_with_tracking(df, svi_variables=DATA[1:], sequence_length=sequence_length)

    # Want to do temporal predictions, so split the data by year
    data_set = yearly_data_split(X_seq, X_static, y, fips_arr, year_arr, year=2016)
    X_train_seq, X_val_seq, X_train_static, X_val_static, y_train, y_val, fips_train, fips_val, year_train, year_val = data_set

    # Sanity check on what years I'm using to train and validate
    print("Training years:", np.unique(year_train))
    print("Validation years:", np.unique(year_val))
    print(f"Using {sequence_length} previous years for prediction.")
    # Dataloaders
    train_ds = MortalityDataset(X_train_seq, X_train_static, y_train)
    val_ds = MortalityDataset(X_val_seq, X_val_static, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # print(f"input_size: {X_seq.shape[2]}")
    # print(f"static_size: {X_static.shape[1]}")
    # Model
    model = LSTMMortalityPredictor(input_size=X_seq.shape[2], hidden_size=64, static_size=X_static.shape[1])
    trained_model = train_model(model, train_loader, val_loader, n_epochs=100, lr=5e-3)
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
    
    # os.makedirs('County Classification\LSTM_MR_Preds', exist_ok=True)
    # results_df.to_csv('County Classification\LSTM_MR_Preds\LSTM_MR_predictions.csv', index=False)
    # print("✅ Saved LSTM predictions to County Classification\LSTM_MR_Preds\LSTM_predictions.csv")
    #plot_residual_and_prediction_hist_by_year(results_df)#, out_dir='County Classification/LSTM_MR_Plots')
    return results_df, trained_model, X_val_seq, X_val_static

def lstm_without_urb(sequence_length=2):
    set_random_seeds()
    #sequence_length = 2  # Number of previous years to use for prediction
    
    print('Running LSTM model for mortality prediction...')
    # Prepare the dataset
    df = prepare_yearly_prediction_data_mortality()
    X_seq, X_static, y, fips_arr, year_arr = prepare_lstm_dataset_with_tracking_v2(df, svi_variables=DATA[1:], sequence_length=sequence_length, include_urbanicity=False)

    # Want to do temporal predictions, so split the data by year
    data_set = yearly_data_split_v2(X_seq, X_static, y, fips_arr, year_arr, year=2016, include_urbanicity=False)
    X_train_seq, X_val_seq, _, _, y_train, y_val, fips_train, fips_val, year_train, year_val = data_set

    # Sanity check on what years I'm using to train and validate
    print("Training years:", np.unique(year_train))
    print("Validation years:", np.unique(year_val))
    print(f"Using {sequence_length} previous years for prediction.")
    # Dataloaders
    train_ds = MortalityDataset_v2(X_train_seq, y_train)
    val_ds = MortalityDataset_v2(X_val_seq, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # print(f"input_size: {X_seq.shape[2]}")
    # print(f"static_size: {X_static.shape[1]}")
    # Model
    model = LSTMMortalityPredictor_v2(input_size=X_seq.shape[2], hidden_size=64,)#, static_size=X_static.shape[1])
    trained_model = train_model_v2(model, train_loader, val_loader, n_epochs=100, lr=5e-3, use_urbanicity=False)
    print('Model training complete.')

    # Get preditions
    y_true, y_pred = get_predictions_v2(trained_model, val_loader, use_urbanicity=False)
    # plot_residual_and_prediction_hist(y_true, y_pred)

    results_df = pd.DataFrame({
    'FIPS': fips_val,
    'Year': year_val,
    'True_MR': y_true,
    'Pred_MR': y_pred
    })
    
    return results_df

### 7/3/25, EB: For completion's sake, the first main function here runs the LSTM model like normal, including the urbanicity category info.

def main():
    ### 6/18/25, EB: Results are below, I found that the best architecture was a 64 hidden size, 2 layers, and a learning rate of 0.0005.
    ### However, a single hidden layer only increased the overall RMSE by 0.1, so we'll go with that for now.
    set_random_seeds()
    sequence_length = 1  # Number of previous years to use for prediction
    
    print('Running LSTM model for mortality prediction...')
    # Prepare the dataset
    df = prepare_yearly_prediction_data_mortality()
    X_seq, X_static, y, fips_arr, year_arr = prepare_lstm_dataset_with_tracking(df, svi_variables=DATA[1:], sequence_length=sequence_length)

    # Want to do temporal predictions, so split the data by year
    data_set = yearly_data_split(X_seq, X_static, y, fips_arr, year_arr, year=2016)
    X_train_seq, X_val_seq, X_train_static, X_val_static, y_train, y_val, fips_train, fips_val, year_train, year_val = data_set

    # Sanity check on what years I'm using to train and validate
    print("Training years:", np.unique(year_train))
    print("Validation years:", np.unique(year_val))
    print(f"Using {sequence_length} previous years for prediction.")
    # Dataloaders
    train_ds = MortalityDataset(X_train_seq, X_train_static, y_train)
    val_ds = MortalityDataset(X_val_seq, X_val_static, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # print(f"input_size: {X_seq.shape[2]}")
    # print(f"static_size: {X_static.shape[1]}")
    # Model
    model = LSTMMortalityPredictor(input_size=X_seq.shape[2], hidden_size=64, static_size=X_static.shape[1])
    trained_model = train_model(model, train_loader, val_loader, n_epochs=100, lr=5e-3)
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
    
    #os.makedirs('County Classification\LSTM_MR_Preds', exist_ok=True)
    #results_df.to_csv('County Classification\LSTM_MR_Preds\LSTM_MR_predictions_threeyear.csv', index=False)
    #print("✅ Saved LSTM predictions to County Classification\LSTM_MR_Preds\LSTM_predictions_threeyear.csv")
    plot_residual_and_prediction_hist_by_year(results_df)#, out_dir='County Classification/LSTM_MR_Plots')

### 7/1/25, EB: Main function to test multiple sequence lengths and compare results:

# def main():
#     set_random_seeds()
#     df = prepare_yearly_prediction_data_mortality()
#     results_by_length = {}

#     for sequence_length in [1, 2, 3]:
#         print(f"\n=== Running LSTM for sequence_length = {sequence_length} ===")
#         # Prepare dataset
#         X_seq, X_static, y, fips_arr, year_arr = prepare_lstm_dataset_with_tracking(
#             df, svi_variables=DATA[1:], sequence_length=sequence_length
#         )

#         if len(X_seq) == 0:
#             print(f"⚠️ No valid sequences for sequence_length = {sequence_length}. Skipping.")
#             continue

#         # Temporal validation split
#         data_set = yearly_data_split(X_seq, X_static, y, fips_arr, year_arr, year=2016)
#         (
#             X_train_seq, X_val_seq,
#             X_train_static, X_val_static,
#             y_train, y_val,
#             fips_train, fips_val,
#             year_train, year_val
#         ) = data_set

#         print("Training years:", np.unique(year_train))
#         print("Validation years:", np.unique(year_val))

#         # Dataloaders
#         train_ds = MortalityDataset(X_train_seq, X_train_static, y_train)
#         val_ds = MortalityDataset(X_val_seq, X_val_static, y_val)
#         train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
#         val_loader   = DataLoader(val_ds, batch_size=64)

#         # Model
#         model = LSTMMortalityPredictor(
#             input_size=X_seq.shape[2],
#             hidden_size=64,
#             static_size=X_static.shape[1]
#         )
#         trained_model = train_model(model, train_loader, val_loader, n_epochs=100, lr=5e-3)
#         print("✅ Model training complete.")

#         # Get predictions
#         y_true, y_pred = get_predictions(trained_model, val_loader)

#         results_df = pd.DataFrame({
#             'FIPS': fips_val,
#             'Year': year_val,
#             'True_MR': y_true,
#             'Pred_MR': y_pred
#         })

#         # Evaluate
#         metrics = evaluate_predictions(results_df)
#         results_by_length[sequence_length] = metrics

#         # Optional: save or plot per-model results
#         plot_residual_and_prediction_hist_by_year(results_df)

#     # Summarize across models
#     summary_df = pd.DataFrame(results_by_length).T
#     print("\n=== Evaluation Summary ===")
#     print(summary_df)


### 7/3/25, EB: Ok, here's another main function that includes a toggle for including the urbanicity class as a static feature or not.
# def main():
#     set_random_seeds()
#     sequence_length = 2  # Number of previous years to use for prediction
    
#     print('Running LSTM model for mortality prediction...')
#     # Prepare the dataset
#     df = prepare_yearly_prediction_data_mortality()
#     X_seq, X_static, y, fips_arr, year_arr = prepare_lstm_dataset_with_tracking_v2(df, svi_variables=DATA[1:], sequence_length=sequence_length, include_urbanicity=False)

#     # Want to do temporal predictions, so split the data by year
#     data_set = yearly_data_split_v2(X_seq, X_static, y, fips_arr, year_arr, year=2016, include_urbanicity=False)
#     X_train_seq, X_val_seq, _, _, y_train, y_val, fips_train, fips_val, year_train, year_val = data_set

#     # Sanity check on what years I'm using to train and validate
#     print("Training years:", np.unique(year_train))
#     print("Validation years:", np.unique(year_val))
#     print(f"Using {sequence_length} previous years for prediction.")
#     # Dataloaders
#     train_ds = MortalityDataset_v2(X_train_seq, y_train)
#     val_ds = MortalityDataset_v2(X_val_seq, y_val)
#     train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=64)

#     # print(f"input_size: {X_seq.shape[2]}")
#     # print(f"static_size: {X_static.shape[1]}")
#     # Model
#     model = LSTMMortalityPredictor_v2(input_size=X_seq.shape[2], hidden_size=64,)#, static_size=X_static.shape[1])
#     trained_model = train_model_v2(model, train_loader, val_loader, n_epochs=100, lr=5e-3, use_urbanicity=False)
#     print('Model training complete.')

#     # Get preditions
#     y_true, y_pred = get_predictions_v2(trained_model, val_loader, use_urbanicity=False)
#     # plot_residual_and_prediction_hist(y_true, y_pred)

#     results_df = pd.DataFrame({
#     'FIPS': fips_val,
#     'Year': year_val,
#     'True_MR': y_true,
#     'Pred_MR': y_pred
#     })
    
#     # os.makedirs('County Classification\LSTM_MR_Preds', exist_ok=True)
#     # results_df.to_csv('County Classification\LSTM_MR_Preds\LSTM_MR_predictions.csv', index=False)
#     # print("✅ Saved LSTM predictions to County Classification\LSTM_MR_Preds\LSTM_predictions.csv")
#     plot_residual_and_prediction_hist_by_year(results_df)#, out_dir='County Classification/LSTM_MR_Plots')


### 7/6/25, EB: The following main function runs the LSTM model with urbanicity, and then without,
### and uses the new plotting function to compare the error histograms laid on top of each other.
# def main():
#     OUT_DIR = 'County Classification/LSTM_Plots'
#     urbanicity_path = 'Data/SVI/NCHS_urban_v_rural.csv'
#     sequence_length = [1]#, 2, 3]  # Number of previous years to use for prediction
#     for seq_length in sequence_length:
#         print(f"\n=== Running LSTM Analysis for sequence_length = {seq_length} ===")
#         results_df_with_urb, trained_model, X_val_seq, X_val_static = lstm_with_urb(sequence_length=seq_length)
#         #results_df_without_urb = lstm_without_urb(sequence_length=seq_length)
#         #print("Comparing models with and without urbanicity...")
#         # plot_comparison_residual_and_prediction_hist_by_year(
#         #     results_df_with_urb,
#         #     results_df_without_urb,
#         #     label_1='With Urbanicity',
#         #     label_2='Without Urbanicity',
#         #     out_dir=OUT_DIR,
#         #     sequence_length=seq_length
#         # )
        
#         ##################
#         ### The following computes the residuals and performs a t-test to determine if 
#         ### the urbanicity information improves the model's performance.
        
#         # abs_error_with  = np.abs(results_df_with_urb['True_MR'] - results_df_with_urb['Pred_MR'])
#         # abs_error_without = np.abs(results_df_without_urb['True_MR'] - results_df_without_urb['Pred_MR'])

#         # # Compute per-county difference
#         # error_diff = abs_error_without - abs_error_with

#         # # Summary stats
#         # print("Mean improvement with urbanicity:", error_diff.mean())

#         # # T-test to see if error reduction is significant
#         # from scipy.stats import ttest_rel
#         # t_stat, p_val = ttest_rel(abs_error_without, abs_error_with)
#         # print(f"Paired t-test p = {p_val:.4f}")


#         ##############
#         ### The following plots the error boxplots by urbanicity class.
#         # print("Plotting error boxplots by urbanicity class...")
#         # plot_error_boxplots_by_urbanicity(
#         #     results_df_with_urb,
#         #     results_df_without_urb,
#         #     urbanicity_path=urbanicity_path
#         # )
        
#         ##############
#         ### The following computes SHAP values for the LSTM model with urbanicity.
#         # shap_values, feature_names = compute_shap_values_for_lstm_with_static(
#         #                                 model=trained_model,
#         #                                 X_seq=X_val_seq,
#         #                                 X_static=X_val_static,
#         #                                 sequence_length=X_val_seq.shape[1],
#         #                                 feature_names_seq=DATA[1:],  # SVI variables
#         #                                 feature_names_static=[f"Urbanicity_{i}" for i in range(X_val_static.shape[1])],
#         #                                 num_samples=100,
#         #                                 background_size=100
#         #                             )
#         # summarize_shap_importance(shap_values, feature_names, top_n=20, plot=True)

#         ##############
#         ### The following plots the error histograms by urbanicity class.
#         print("Plotting error histograms by urbanicity class...")
#         results_with_urban = attach_urbanicity(results_df_with_urb, urbanicity_path=urbanicity_path)
#         plot_error_histograms_by_urbanicity(results_with_urban, bins=40)


    
if __name__ == "__main__":
    main()




###################################################################################################################################

### 6/18/25, EB: I ran the grid search over architectures and learning rates, and found the following results:
# 📊 Architecture × LR comparison (sorted by RMSE):
#     hidden_size  num_layers      lr    val_mae   val_rmse
# 0            64           2  0.0005  11.122872  17.453268
# 1            64           2  0.0010  11.002099  17.458107
# 2           128           2  0.0020  10.903929  17.465984
# 3            64           1  0.0005  11.173127  17.581680
# 4           128           1  0.0005  11.110516  17.610186
# 5            64           2  0.0020  11.028882  17.691069
# 6            32           1  0.0020  11.226838  17.750135
# 7           128           1  0.0020  11.141236  17.775858
# 8            32           1  0.0005  11.338222  17.849943
# 9           128           2  0.0001  11.529820  17.900921
# 10           32           1  0.0001  11.576588  17.947050
# 11           64           1  0.0001  11.568151  17.973013
# 12           32           1  0.0010  11.328876  17.985537
# 13           64           2  0.0001  11.723288  18.104595
# 14          128           1  0.0001  11.678407  18.159500
# 15           64           1  0.0020  21.328650  29.588905
# 16          128           2  0.0010  21.328650  29.588905
# 17          128           1  0.0010  21.328650  29.588905
# 18          128           2  0.0005  21.328650  29.588905
# 19           64           1  0.0010  21.328650  29.588905


### The following are scripts to find the best architecture and learning rate for the LSTM model.
# ### 6/17/25, EB: The following chunk runs through several different architectures to compare performance.
# ### The results showed that it made little difference, so below I'm including a grid-search over learning rate as well.
# # 📊 Architecture Comparison:
# #    hidden_size  num_layers    val_mae   val_rmse
# # 4          128           2  10.821910  17.385542
# # 2           64           2  10.942174  17.459635
# # 1           64           1  11.056852  17.565632
# # 3          128           1  11.091640  17.738924
# # 0           32           1  11.298776  17.829168
#     # print("\n🔍 Running architecture comparison...")
#     # architectures = [
#     #     {'hidden_size': 32, 'num_layers': 1},
#     #     {'hidden_size': 64, 'num_layers': 1},
#     #     {'hidden_size': 64, 'num_layers': 2},
#     #     {'hidden_size': 128, 'num_layers': 1},
#     #     {'hidden_size': 128, 'num_layers': 2},
#     # ]

#     # results = []

#     # for config in architectures:
#     #     print(f"\n🔧 Training model with hidden_size={config['hidden_size']} | num_layers={config['num_layers']}")
        
#     #     # Build model
#     #     model = LSTMMortalityPredictor(
#     #         input_size=X_train_seq.shape[2],
#     #         hidden_size=config['hidden_size'],
#     #         static_size=X_train_static.shape[1],
#     #         num_layers=config['num_layers'],
#     #         dropout=0.2
#     #     )

#     #     # Train model
#     #     trained_model = train_model(model, train_loader, val_loader, n_epochs=100, lr=1e-3)

#     #     # Get predictions
#     #     y_true, y_pred = get_predictions(trained_model, val_loader)

#     #     # Evaluate
#     #     mae = mean_absolute_error(y_true, y_pred)
#     #     rmse = mean_squared_error(y_true, y_pred, squared=False)

#     #     print(f"✅ Validation MAE: {mae:.2f} | RMSE: {rmse:.2f}")

#     #     results.append({
#     #         'hidden_size': config['hidden_size'],
#     #         'num_layers': config['num_layers'],
#     #         'val_mae': mae,
#     #         'val_rmse': rmse
#     #     })

#     # # View results
#     # results_df = pd.DataFrame(results).sort_values('val_rmse')
#     # print("\n📊 Architecture Comparison:")
#     # print(results_df)
    
    
# ### 6/17/25, EB: Now let's do a grid search over learning rates AND architectures.
#     # ── hyper-configs ───────────────────────────────────────────────────────────────
#     architectures = [
#         {'hidden_size': 32,  'num_layers': 1},
#         {'hidden_size': 64,  'num_layers': 1},
#         {'hidden_size': 64,  'num_layers': 2},
#         {'hidden_size': 128, 'num_layers': 1},
#         {'hidden_size': 128, 'num_layers': 2},
#     ]

#     learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]           # ← add / tweak as you wish
#     n_epochs       = 100                                # keep or tune

#     # ── run sweep ───────────────────────────────────────────────────────────────────
#     results = []

#     for cfg in architectures:
#         for lr in learning_rates:
#             print(f"\n🔧 hidden={cfg['hidden_size']}  layers={cfg['num_layers']}  lr={lr:g}")

#             # build model
#             model = LSTMMortalityPredictor(
#                 input_size  = X_train_seq.shape[2],
#                 hidden_size = cfg['hidden_size'],
#                 static_size = X_train_static.shape[1],
#                 num_layers  = cfg['num_layers'],
#                 dropout     = 0.2
#             )

#             # train
#             trained_model = train_model(
#                 model, train_loader, val_loader,
#                 n_epochs=n_epochs, lr=lr
#             )

#             # evaluate
#             y_true, y_pred = get_predictions(trained_model, val_loader)
#             mae  = mean_absolute_error(y_true, y_pred)
#             rmse = mean_squared_error(y_true, y_pred, squared=False)

#             print(f"   ►  MAE={mae:.2f}  RMSE={rmse:.2f}")

#             results.append({
#                 'hidden_size': cfg['hidden_size'],
#                 'num_layers' : cfg['num_layers'],
#                 'lr'         : lr,
#                 'val_mae'    : mae,
#                 'val_rmse'   : rmse
#             })

#     # ── leaderboard ────────────────────────────────────────────────────────────────
#     results_df = (
#         pd.DataFrame(results)
#         .sort_values('val_rmse')
#         .reset_index(drop=True)
#     )

#     print("\n📊 Architecture × LR comparison (sorted by RMSE):")
#     print(results_df)






