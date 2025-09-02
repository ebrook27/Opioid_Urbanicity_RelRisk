### 7/15/25, EB: Ok, I talked with Adam and Andrew on Thursday (7/10) and Adam suggested we try to predict the percentile ranking
### of the counties, rather than the actual mortality rate. This way, we can see how each county compares to others in the same year.
### I'll have to fit a log-normal distribution to the mortality rates for each year, in order to evaluate the predictions.
### Adam also suggested we finally try to include the social capital and connectedness data as input features.
### This will be require some careful consideration, since these data are only available for single years. We'll likely have to use a similar
### approach as with the urbanicity category, including them as static features in a multi-modal sort of model.
### My first priority is to get the percentile ranking predictions working, and then expand the input data from there.

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

from scipy.stats import norm
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    ### ------ SVI variables already normalized ------ ###
    # # Normalize SVI variables individually
    # scalers = {var: StandardScaler() for var in svi_variables}
    # for var in svi_variables:
    #     df_encoded[var] = scalers[var].fit_transform(df_encoded[[var]])

    ### ⬇️ INSERT LOG-NORMAL PERCENTILE RANKING HERE
    # Handle zeros before log transform
    df_encoded['mortality_next'] = df_encoded['mortality_next'].replace(0, np.nan)
    df_encoded = df_encoded.dropna(subset=['mortality_next'])

    # Compute log-normal percentile ranks year by year
    for year, group in df_encoded.groupby('year'):
        log_vals = np.log(group['mortality_next'].values)
        mu, sigma = np.mean(log_vals), np.std(log_vals)
        percentiles = norm.cdf(np.log(group['mortality_next'].values), loc=mu, scale=sigma) #* 100
        df_encoded.loc[group.index, 'percentile_rank'] = percentiles
    ### ⬆️ END INSERT

    # Now prepare sequences
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
        percentile_seq = group['percentile_rank'].values
        years = group['year'].values

        for i in range(len(group) - sequence_length):
            x_seq = svi_seq[i:i+sequence_length]
            y_target = percentile_seq[i + sequence_length]
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
#     def __init__(self, input_size, hidden_size, static_size, num_layers=1, dropout=0.0):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size + static_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()  # Ensure output stays in [0, 1]
#         )

#     def forward(self, x_seq, x_static):
#         lstm_out, _ = self.lstm(x_seq)
#         last_output = lstm_out[:, -1, :]
#         combined = torch.cat((last_output, x_static), dim=1)
#         return self.fc(combined).squeeze(1)  # Still returns a shape of (batch,)

### 7/16/25, EB: Pretty poor performance, going to try a different model architecture, including dropout and more layers.
class LSTMMortalityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, static_size, num_layers=1, dropout=0.3):
        super().__init__()

        # LSTM with dropout (only applies when num_layers > 1)
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # FC head with dropout + batch norm + ReLU
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Ensures output is in [0, 1] range
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        last_output = lstm_out[:, -1, :]  # Last time step
        combined = torch.cat((last_output, x_static), dim=1)
        return self.fc(combined).squeeze(1)


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, preds, targets):
        errors = targets - preds
        loss = torch.max(
            (self.quantile - 1) * errors,
            self.quantile * errors
        )
        return torch.mean(loss)


def train_model(model, train_loader, val_loader, n_epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()  # More robust to outliers than MSE
    #criterion = QuantileLoss(quantile=0.5)  # Using quantile loss for better percentile prediction
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

    y_true = np.array(y_true) * 100  # Convert to percentage
    y_pred = np.array(y_pred) * 100  # Convert to percentage

    return y_true, y_pred

# def plot_residual_and_prediction_hist_by_year(results_df, out_dir=None):
#     """
#     Plots a figure for each year in the results dataframe.
#     Each figure has:
#       - Left: Histogram of residuals (absolute error)
#       - Right: Histogram of predicted mortality rates

#     If out_dir is provided, saves each plot instead of showing it.
#     """
#     years = sorted(results_df['Year'].unique())
    
#     for year in years:
#         df_year = results_df[results_df['Year'] == year].copy()
#         residuals = np.abs(df_year['Pred_MR_Percentile'] - df_year['True_MR_Percentile'])
#         preds = df_year['Pred_MR_Percentile']

#         fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#         # Residuals
#         sns.histplot(residuals, bins=50, kde=False, ax=axes[0], color='skyblue')
#         axes[0].set_title(f"{year} Residuals (|Pred - True| MR)")
#         axes[0].set_xlabel("Absolute Error")
#         axes[0].set_ylabel("Frequency")
#         axes[0].grid(True)

#         # Predictions
#         sns.histplot(preds, bins=50, kde=False, ax=axes[1], color='salmon')
#         axes[1].set_title(f"{year} Predicted Mortality Rate Distribution")
#         axes[1].set_xlabel("Predicted Mortality Rate")
#         axes[1].set_ylabel("Frequency")
#         axes[1].grid(True)

#         plt.tight_layout()

#         if out_dir:
#             os.makedirs(out_dir, exist_ok=True)
#             plt.savefig(f"{out_dir}/residual_and_prediction_hist_{year}.png", dpi=300, bbox_inches='tight')
#             plt.close()
#         else:
#             plt.show()

def plot_percentile_prediction_performance(results_df, sequence_length, out_dir=None):
    """
    For each year, plots:
      - Histogram of absolute errors (percentile space)
      - Scatter plot of predicted vs. true percentile
      - Boxplot: predicted percentiles by true percentile decile group
    """
    years = sorted(results_df['Year'].unique())

    for year in years:
        df_year = results_df[results_df['Year'] == year].copy()
        df_year['Abs_Error'] = np.abs(df_year['Pred_Percentile'] - df_year['True_Percentile'])

        # Bin true percentiles into deciles for boxplot
        df_year['True_Bin'] = pd.qcut(df_year['True_Percentile'], q=10, labels=[f'D{d}' for d in range(1,11)])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Histogram of absolute error
        sns.histplot(df_year['Abs_Error'], bins=40, ax=axes[0], color='skyblue')
        axes[0].set_title(f'{year} Absolute Error (|Pred - True|)')
        axes[0].set_xlabel('Absolute Error (Percentile Points)')
        axes[0].set_ylabel('Count')

        # 2. Scatter of true vs predicted percentiles
        sns.scatterplot(data=df_year, x='True_Percentile', y='Pred_Percentile', alpha=0.3, ax=axes[1])
        axes[1].plot([0, 100], [0, 100], ls='--', c='gray')
        axes[1].set_title(f'{year} Predicted vs True Percentile')
        axes[1].set_xlabel('True Percentile')
        axes[1].set_ylabel('Predicted Percentile')

        # 3. Boxplot: predicted percentiles by decile of true percentile
        sns.boxplot(data=df_year, x='True_Bin', y='Pred_Percentile', ax=axes[2], palette='pastel')
        axes[2].set_title(f'{year} Predicted Percentiles by True Decile')
        axes[2].set_xlabel('True Percentile Decile')
        axes[2].set_ylabel('Predicted Percentile')

        plt.suptitle(f"Mortality Rate Distribution and Percentile Comparison, {sequence_length} Prior Years ({year})", fontsize=14)
        plt.tight_layout()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/{sequence_length}_percentile_perf_{year}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():

    set_random_seeds(42)
    sequence_length = [1,2,3]  # Number of previous years to use for prediction
    
    for seq_length in sequence_length:
        print(f"Running LSTM model for mortality prediction with {seq_length} year sequence...")
        # print('Running LSTM model for mortality prediction...')
        # Prepare the dataset
        df = prepare_yearly_prediction_data_mortality()
        X_seq, X_static, y, fips_arr, year_arr = prepare_lstm_dataset_with_tracking(df, svi_variables=DATA[1:], sequence_length=seq_length)

        # Want to do temporal predictions, so split the data by year
        data_set = yearly_data_split(X_seq, X_static, y, fips_arr, year_arr, year=2016)
        X_train_seq, X_val_seq, X_train_static, X_val_static, y_train, y_val, fips_train, fips_val, year_train, year_val = data_set

        # Sanity check on what years I'm using to train and validate
        print("Training years:", np.unique(year_train))
        print("Validation years:", np.unique(year_val))
        print(f"Using {seq_length} previous years for prediction.")
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

        # Get predictions
        y_true, y_pred = get_predictions(trained_model, val_loader)
        # plot_residual_and_prediction_hist(y_true, y_pred)

        results_df = pd.DataFrame({
        'FIPS': fips_val,
        'Year': year_val,
        'True_Percentile': y_true,
        'Pred_Percentile': y_pred
        })
        
        #os.makedirs('County Classification\LSTM_MR_Preds', exist_ok=True)
        #results_df.to_csv('County Classification\LSTM_MR_Preds\LSTM_MR_predictions_threeyear.csv', index=False)
        #print("✅ Saved LSTM predictions to County Classification\LSTM_MR_Preds\LSTM_predictions_threeyear.csv")
        plot_percentile_prediction_performance(results_df, sequence_length=seq_length, out_dir='County Classification/LSTM_Percentile_Plots')

if __name__ == "__main__":
    main()