### 7/17/25, EB: Not really sure what to dub this approach, but in the file EB_LSTM_Percentile_Model.py, I used an LSTM
### to try to predict the percentile rank of mortality rates for several years in the study. Even using 3 previous years to predict,
### the model performed pretty poorly, with a lot of variance in the predictions.
### OTOH, using the LSTM to predict the mortality rate directly (as in EB_LSTM_Mortality_Model.py) yielded much better results.
### Adam wants percetnile rankings, though, so I had the idea of fitting a Log-Normal distribution to the predicted mortality rates
### and then using the CDF to get the percentile rank. We can compare the predicted percentile ranks to the actual ranks
### and my hope is that this will give us good results.
### So, I'm going to copy over the original working model and pipeline from EB_LSTM_Mortality_Model.py, and then modify it to 
### work with the percentile ranks instead of the mortality rates.


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

from scipy.stats import lognorm
import numpy as np
import pandas as pd

def evaluate_log_normal_percentiles(results_df):
    """
    Fits log-normal distributions to true and predicted mortality rates for each year,
    computes percentile ranks using the fitted distributions, and evaluates errors.
    Returns updated dataframe and summary metrics.
    """

    results_df = results_df.copy()
    pred_percentiles = []
    true_percentiles = []
    summary = []

    for year in sorted(results_df['Year'].unique()):
        df_year = results_df[results_df['Year'] == year]
        true_vals = df_year['True_MR'].values
        pred_vals = df_year['Pred_MR'].values

        # Clip to avoid issues with log-normal fitting
        true_vals_clipped = np.clip(true_vals, 1e-5, None)
        pred_vals_clipped = np.clip(pred_vals, 1e-5, None)

        # Fit log-normal to actual and predicted mortality
        shape_t, loc_t, scale_t = lognorm.fit(true_vals_clipped, floc=0)
        shape_p, loc_p, scale_p = lognorm.fit(pred_vals_clipped, floc=0)

        # Compute percentile ranks using CDF
        year_true_pct = lognorm.cdf(true_vals_clipped, s=shape_t, loc=loc_t, scale=scale_t) * 100
        year_pred_pct = lognorm.cdf(pred_vals_clipped, s=shape_p, loc=loc_p, scale=scale_p) * 100

        pred_percentiles.extend(year_pred_pct)
        true_percentiles.extend(year_true_pct)

        # Evaluation metrics
        mae = np.mean(np.abs(year_pred_pct - year_true_pct))
        mse = np.mean((year_pred_pct - year_true_pct) ** 2)

        summary.append({'Year': year, 'MAE_percentile': mae, 'MSE_percentile': mse})

    # Add percentiles to the dataframe
    results_df['True_MR_Percentile'] = true_percentiles
    results_df['Pred_MR_Percentile'] = pred_percentiles

    print("📊 Log-normal percentile evaluation summary:")
    for row in summary:
        print(f"{row['Year']}: MAE={row['MAE_percentile']:.2f}, MSE={row['MSE_percentile']:.2f}")

    return results_df, pd.DataFrame(summary)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm
import numpy as np
import os

# def plot_predicted_vs_actual_distributions(results_df, seq_length, out_dir=None):
#     """
#     For each year:
#       - KDE overlay of actual and predicted MR
#       - Log-normal PDF fits for both
#       - Scatterplot of predicted vs actual percentile ranks
#     Saves or displays plots per year.
#     """
#     years = sorted(results_df['Year'].unique())

#     for year in years:
#         df_year = results_df[results_df['Year'] == year].copy()
#         true_vals = df_year['True_MR'].clip(lower=1e-5)
#         pred_vals = df_year['Pred_MR'].clip(lower=1e-5)

#         # Fit log-normal distributions
#         shape_t, loc_t, scale_t = lognorm.fit(true_vals, floc=0)
#         shape_p, loc_p, scale_p = lognorm.fit(pred_vals, floc=0)

#         # Range for plotting PDFs
#         x = np.linspace(min(true_vals.min(), pred_vals.min()), 
#                         max(true_vals.max(), pred_vals.max()), 500)
#         pdf_true = lognorm.pdf(x, s=shape_t, loc=loc_t, scale=scale_t)
#         pdf_pred = lognorm.pdf(x, s=shape_p, loc=loc_p, scale=scale_p)

#         # Compute percentile ranks
#         actual_percentiles = lognorm.cdf(true_vals, s=shape_t, loc=loc_t, scale=scale_t) * 100
#         pred_percentiles   = lognorm.cdf(pred_vals, s=shape_p, loc=loc_p, scale=scale_p) * 100

#         # Start plotting
#         fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#         # 1. KDE Overlay
#         sns.kdeplot(true_vals, label='Actual', color='blue', fill=True, alpha=0.3, ax=axes[0])
#         sns.kdeplot(pred_vals, label='Predicted', color='red', fill=True, alpha=0.3, ax=axes[0])
#         axes[0].set_title("KDE of Mortality Rates")
#         axes[0].set_xlabel("Mortality Rate")
#         axes[0].set_ylabel("Density")
#         axes[0].legend()
#         axes[0].grid(True)

#         # 2. Log-normal PDF fits
#         axes[1].plot(x, pdf_true, color='blue', label='LogNorm Fit (Actual)', linestyle='--')
#         axes[1].plot(x, pdf_pred, color='red', label='LogNorm Fit (Predicted)', linestyle='--')
#         axes[1].set_title("Fitted Log-Normal PDFs")
#         axes[1].set_xlabel("Mortality Rate")
#         axes[1].set_ylabel("PDF")
#         axes[1].legend()
#         axes[1].grid(True)

#         # 3. Percentile rank scatterplot
#         axes[2].scatter(actual_percentiles, pred_percentiles, alpha=0.5, color='purple', edgecolor='white', s=40)
#         axes[2].plot([0, 100], [0, 100], 'k--', linewidth=1)
#         axes[2].set_xlim(0, 100)
#         axes[2].set_ylim(0, 100)
#         axes[2].set_title("Percentile Rank: Predicted vs Actual")
#         axes[2].set_xlabel("Actual Percentile")
#         axes[2].set_ylabel("Predicted Percentile")
#         axes[2].grid(True)

#         plt.suptitle(f"Mortality Rate Distribution and Percentile Comparison, {seq_length} Prior Years ({year})", fontsize=14)
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#         if out_dir:
#             os.makedirs(out_dir, exist_ok=True)
#             plt.savefig(f"{out_dir}/{seq_length}year_seqlength_MR_comparison_{year}.png", dpi=300, bbox_inches='tight')
#             plt.close()
#         else:
#             plt.show()

def plot_predicted_vs_actual_distributions(results_df, seq_length, out_dir=None):
    """
        For each year:
        - KDE overlay of actual and predicted MR
        - Scatterplot of predicted vs actual percentile ranks
        Saves or displays plots per year.
    """
    years = sorted(results_df['Year'].unique())

    for year in years:
        df_year = results_df[results_df['Year'] == year].copy()
        true_vals = df_year['True_MR'].clip(lower=1e-5)
        pred_vals = df_year['Pred_MR'].clip(lower=1e-5)

        # Fit log-normal distributions
        shape_t, loc_t, scale_t = lognorm.fit(true_vals, floc=0)
        shape_p, loc_p, scale_p = lognorm.fit(pred_vals, floc=0)

        # Compute percentile ranks
        actual_percentiles = lognorm.cdf(true_vals, s=shape_t, loc=loc_t, scale=scale_t) * 100
        pred_percentiles   = lognorm.cdf(pred_vals, s=shape_p, loc=loc_p, scale=scale_p) * 100

        # Start plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. KDE Overlay
        sns.kdeplot(true_vals, label='Actual', color='blue', fill=True, alpha=0.3, ax=axes[0])
        sns.kdeplot(pred_vals, label='Predicted', color='red', fill=True, alpha=0.3, ax=axes[0])
        axes[0].set_title("KDE of Mortality Rates")
        axes[0].set_xlabel("Mortality Rate")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True)

        # 2. Percentile rank scatterplot
        axes[1].scatter(actual_percentiles, pred_percentiles, alpha=0.5, color='purple', edgecolor='white', s=40)
        axes[1].plot([0, 100], [0, 100], 'k--', linewidth=1)
        axes[1].set_xlim(0, 100)
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Percentile Rank: Predicted vs Actual")
        axes[1].set_xlabel("Actual Percentile")
        axes[1].set_ylabel("Predicted Percentile")
        axes[1].grid(True)

        plt.suptitle(f"Mortality Rate Comparison and Rank Accuracy, {seq_length} Prior Years ({year})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/{seq_length}year_seqlength_MR_comparison_{year}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    ### 6/18/25, EB: Results are below, I found that the best architecture was a 64 hidden size, 2 layers, and a learning rate of 0.0005.
    ### However, a single hidden layer only increased the overall RMSE by 0.1, so we'll go with that for now.
    set_random_seeds()
    sequence_length = [1,2,3]  # Number of previous years to use for prediction
    
    for seq_length in sequence_length:
        print(f"Running LSTM model for mortality prediction with {seq_length} year sequence...")
        #print('Running LSTM model for mortality prediction...')
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
        # results_df.to_csv('County Classification\LSTM_MR_Preds\LSTM_MR_predictions_threeyear.csv', index=False)
        # print("✅ Saved LSTM predictions to County Classification\LSTM_MR_Preds\LSTM_predictions_threeyear.csv")
        # plot_residual_and_prediction_hist_by_year(results_df)#, out_dir='County Classification/LSTM_MR_Plots')
        updated_results_df, summary_df = evaluate_log_normal_percentiles(results_df)
        
        print("📊 Log-normal percentile evaluation summary:")
        for row in summary_df.itertuples(index=False):
            print(f"{row.Year}: MAE={row.MAE_percentile:.2f}, MSE={row.MSE_percentile:.2f}")
        
        plot_predicted_vs_actual_distributions(results_df, seq_length, out_dir=f'Distribution_Comparisons/{seq_length}Year_Sequence')



if __name__ == "__main__":
    main()