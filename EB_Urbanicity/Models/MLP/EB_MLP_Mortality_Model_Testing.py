### 6/4/25, EB: The Random Forest models did what we wanted them to, we found that overall the urbanicity categories were of reasonable importanve,
### and when we predicted within each category we got interesting feature importance rankings.
### What I'm trying to do here is to use a Multi-Layer Perceptron (MLP) to predict mortality, but within each urbanicity category we use the rankings from the 
### RF models to weight the input features. This way we have an "informed" weighting for each category, and the hope is that the MLP will learn to use these features more effectively.


### 6/12/25, EB: Here I'm trying to get predictions for more than just 2021, but I'm having a lot of trouble getting it to work. I had a meeting deadline
### so I copied the experimental code here to keep experimenting, but I didn't want to mess with the original code that was working.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# DATA = ['Mortality',
#         'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
#         'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
#         'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
#         'Single-Parent Household', 'Unemployment']

FEATURE_IMPORTANCE_CSV = 'County Classification/RF_Mort_Preds_by_Urbanicity/RF_Feat_Imp/RF_feature_importances.csv'
RESULTS_DIR = 'County Classification/MLP_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze()

# def prepare_yearly_prediction_data():
#     # (Use your existing function body unchanged)
#     svi_variables = [v for v in DATA if v != 'Mortality']
#     years = list(range(2010, 2022))
#     nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
#     nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
#     nchs_df = nchs_df.set_index('FIPS')
#     nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)
#     mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
#     mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
#     mort_df = mort_df.set_index('FIPS')
#     svi_data = []
#     for var in svi_variables:
#         var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
#         var_df = pd.read_csv(var_path, dtype={'FIPS': str})
#         var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
#         long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)
#         long_df['year'] = long_df['year_var'].str.extract(r'(\\d{4})').astype(int)
#         long_df = long_df[long_df['year'].between(2010, 2021)]
#         long_df = long_df.drop(columns='year_var')
#         svi_data.append(long_df)
#     from functools import reduce
#     svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)
#     for y in years:
#         mort_col = f'{y+1} MR'
#         if mort_col not in mort_df.columns:
#             continue
#         svi_merged.loc[svi_merged['year'] == y, 'mortality_rate'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(mort_df[mort_col])
#     svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')
#     svi_merged = svi_merged.dropna()
#     return svi_merged

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

def prepare_yearly_prediction_data():
    """Creates a long-format dataset for predicting next-year mortality using current-year SVI + county class."""
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2022))  # We predict mortality up to 2022

    # Load county category (static)
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)
    
    # Load mortality
    mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
    mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
    mort_df = mort_df.set_index('FIPS')
    
    # Load all SVI variables and reshape to long format per county-year
    svi_data = []
    for var in svi_variables:
        var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
        var_df = pd.read_csv(var_path, dtype={'FIPS': str})
        var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
        long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)

        # Extract year and filter relevant years
        long_df['year'] = long_df['year_var'].str.extract(r'(\d{4})').astype(int)
        long_df = long_df[long_df['year'].between(2010, 2023)]  # we predict 1 year ahead
        long_df = long_df.drop(columns='year_var')
        svi_data.append(long_df)

    # Merge all SVI variables on FIPS + year
    from functools import reduce
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Add mortality for year+1
    for y in years:
        mort_col = f'{y+1} MR'
        if mort_col not in mort_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'mortality_rate'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(mort_df[mort_col])

    # Add county class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    print(svi_merged['year'].value_counts().sort_index())

    # Drop rows with any missing values
    svi_merged = svi_merged.dropna()

    return svi_merged


def load_feature_importances():
    fi_df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    grouped = fi_df.groupby(['county_class', 'Feature'])['Importance'].mean().reset_index()
    grouped['Importance'] = grouped.groupby('county_class')['Importance'].transform(lambda x: x / x.sum())
    return grouped

def apply_feature_weights(df, feature_weights, county_class):
    df_copy = df.copy()
    features = [f for f in df.columns if f in DATA and f != 'Mortality']
    for feature in features:
        weight = feature_weights.loc[
            (feature_weights['county_class'] == county_class) & 
            (feature_weights['Feature'] == feature),
            'Importance'
        ]
        if not weight.empty:
            df_copy[feature] = df_copy[feature] * weight.values[0]
    return df_copy

def train_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=100, lr=0.001):
    model = SimpleMLP(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_tensor).cpu().numpy()
    return y_pred

# def run_mlp_pipeline():
#     df = prepare_yearly_prediction_data()
#     feature_importances = load_feature_importances()

#     results = []
#     county_classes = sorted(df['county_class'].unique())

#     for county_class in county_classes:
#         print(f'🚀 Training PyTorch MLP for County Class: {county_class}')
#         df_sub = df[df['county_class'] == county_class].copy()

#         for year in range(2010, 2022):
#             df_year = df_sub[df_sub['year'] == year].copy()
#             if df_year.empty:
#                 continue

#             df_year = apply_feature_weights(df_year, feature_importances, county_class)
#             X = df_year.drop(columns=['FIPS', 'year', 'mortality_rate', 'county_class'])
#             y = df_year['mortality_rate']

#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X)
#             input_dim = X_scaled.shape[1]

#             kf = KFold(n_splits=5, shuffle=True, random_state=42)
#             for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
#                 X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
#                 y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#                 y_pred = train_mlp(X_train, y_train, X_test, y_test, input_dim)

#                 rmse = mean_squared_error(y_test, y_pred, squared=False)
#                 mae = mean_absolute_error(y_test, y_pred)
#                 r2 = r2_score(y_test, y_pred)

#                 results.append({
#                     'county_class': county_class,
#                     'year': year,
#                     'fold': fold_idx + 1,
#                     'RMSE': rmse,
#                     'MAE': mae,
#                     'R2': r2
#                 })

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(os.path.join(RESULTS_DIR, 'PyTorch_MLP_model_results.csv'), index=False)
#     print(f'✅ Saved PyTorch MLP results to {RESULTS_DIR}/PyTorch_MLP_model_results.csv')


# def run_mlp_pipeline():
#     df = prepare_yearly_prediction_data()
#     feature_importances = load_feature_importances()

#     results = []
#     county_classes = sorted(df['county_class'].unique())

#     all_fold_results = []

#     for county_class in county_classes:
#         print(f'🚀 Training PyTorch MLP for County Class: {county_class}')
#         df_sub = df[df['county_class'] == county_class].copy()

#         for year in range(2010, 2022):
#             df_year = df_sub[df_sub['year'] == year].copy()
#             if df_year.empty:
#                 continue

#             df_year = apply_feature_weights(df_year, feature_importances, county_class)
#             X = df_year.drop(columns=['FIPS', 'year', 'mortality_rate', 'county_class'])
#             y = df_year['mortality_rate']

#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X)
#             input_dim = X_scaled.shape[1]

#             kf = KFold(n_splits=5, shuffle=True, random_state=42)
#             for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
#                 X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
#                 y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#                 y_pred = train_mlp(X_train, y_train, X_test, y_test, input_dim)

#                 rmse = mean_squared_error(y_test, y_pred, squared=False)
#                 mae = mean_absolute_error(y_test, y_pred)
#                 r2 = r2_score(y_test, y_pred)

#                 results.append({
#                     'county_class': county_class,
#                     'year': year,
#                     'fold': fold_idx + 1,
#                     'RMSE': rmse,
#                     'MAE': mae,
#                     'R2': r2
#                 })

#                 # Save per-county predictions
#                 fold_results = pd.DataFrame({
#                     'FIPS': df_year.iloc[test_idx]['FIPS'].values,
#                     'county_class': county_class,
#                     'year': year,
#                     'fold': fold_idx + 1,
#                     'True': y_test.values,
#                     'Predicted': y_pred,
#                     'Absolute_Error': np.abs(y_test.values - y_pred)
#                 })
#                 all_fold_results.append(fold_results)

#     results_df = pd.DataFrame(results)
#     all_fold_results_df = pd.concat(all_fold_results, ignore_index=True)
    
#     results_df.to_csv(os.path.join(RESULTS_DIR, 'PyTorch_MLP_model_results.csv'), index=False)
#     all_fold_results_df.to_csv(os.path.join(RESULTS_DIR, 'PyTorch_MLP_model_fold_preds.csv'), index=False)
    
#     print(f'✅ Saved PyTorch MLP results to {RESULTS_DIR}/PyTorch_MLP_model_results.csv')
#     print(f'✅ Saved PyTorch MLP fold predictions to {RESULTS_DIR}/PyTorch_MLP_model_fold_preds.csv')

def apply_feature_weights_per_row(df, feature_weights):
    df_copy = df.copy()
    features = [f for f in df.columns if f in DATA and f != 'Mortality']

    for idx, row in df_copy.iterrows():
        county_class = row['county_class']
        for feature in features:
            weight = feature_weights.loc[
                (feature_weights['county_class'] == county_class) &
                (feature_weights['Feature'] == feature),
                'Importance'
            ]
            if not weight.empty:
                df_copy.at[idx, feature] *= weight.values[0]
    return df_copy


def run_mlp_pipeline_yearly():
    df = prepare_yearly_prediction_data()
    feature_importances = load_feature_importances()

    results = []
    all_fold_results = []

    for year in range(2010, 2022):
        df_year = df[df['year'] == year].copy()
        if df_year.empty:
            continue

        print(f'🚀 Training PyTorch MLP for Year: {year}')

        # Apply per-class feature weights
        df_year = apply_feature_weights_per_row(df_year, feature_importances)

        X = df_year.drop(columns=['FIPS', 'year', 'mortality_rate', 'county_class'])
        y = df_year['mortality_rate']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            y_pred = train_mlp(X_train, y_train, X_test, y_test, input_dim)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'year': year,
                'fold': fold_idx + 1,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

            fold_results = pd.DataFrame({
                'FIPS': df_year.iloc[test_idx]['FIPS'].values,
                'year': year,
                'fold': fold_idx + 1,
                'True': y_test.values,
                'Predicted': y_pred,
                'Absolute_Error': np.abs(y_test.values - y_pred)
            })
            all_fold_results.append(fold_results)

    results_df = pd.DataFrame(results)
    all_fold_results_df = pd.concat(all_fold_results, ignore_index=True)

    results_df.to_csv(os.path.join(RESULTS_DIR, 'PyTorch_MLP_model_results.csv'), index=False)
    all_fold_results_df.to_csv(os.path.join(RESULTS_DIR, 'PyTorch_MLP_model_fold_preds.csv'), index=False)
    print(f'✅ Saved PyTorch MLP results to {RESULTS_DIR}/PyTorch_MLP_model_results.csv')
    print(f'✅ Saved PyTorch MLP fold predictions to {RESULTS_DIR}/PyTorch_MLP_model_fold_preds.csv')

def run_mlp_pipeline_multiyear_train():
    """
    Train MLP on multiple years of data, predicting mortality for the next year.
    This differs from run_mlp_pipeline_yearly in that it uses a training set spanning multiple years,
    rather than running a k-fold cross-validation for each year individually.
    """

    df = prepare_yearly_prediction_data()
    feature_importances = load_feature_importances()

    results = []
    all_results = []

    # Split training and testing data
    train_years = list(range(2010, 2021))
    test_years = [2021, 2022]  # Test on the last two years

    df_train = df[df['year'].isin(train_years)].copy()
    df_train = apply_feature_weights_per_row(df_train, feature_importances)
    #df_test = df[df['year'].isin(test_years)].copy()

    print(f'🚀 Training PyTorch MLP on Years: {train_years} and testing on {test_years}')

    # # Apply per-class feature weights
    # df_train = apply_feature_weights_per_row(df_train, feature_importances)
    # df_test = apply_feature_weights_per_row(df_test, feature_importances)

    # # Prepare inputs
    X_train = df_train.drop(columns=['FIPS', 'year', 'mortality_rate', 'county_class'])
    y_train = df_train['mortality_rate']
    # X_test = df_test.drop(columns=['FIPS', 'year', 'mortality_rate', 'county_class'])
    # y_test = df_test['mortality_rate']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    input_dim = X_train_scaled.shape[1]

    # # Train the model
    # y_pred = train_mlp(X_train_scaled, y_train, X_test_scaled, y_test, input_dim)

    # # Evaluate performance
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    # mae = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # results.append({
    #     'train_years': f"{min(train_years)}-{max(train_years)}",
    #     'test_years': ','.join(map(str, test_years)),
    #     'RMSE': rmse,
    #     'MAE': mae,
    #     'R2': r2
    # })

    # # Save per-county predictions
    # fold_results = pd.DataFrame({
    #     'FIPS': df_test['FIPS'].values,
    #     'year': df_test['year'].values,
    #     'True': y_test.values,
    #     'Predicted': y_pred,
    #     'Absolute_Error': np.abs(y_test.values - y_pred)
    # })
    # all_results.append(fold_results)

    for year in test_years:
        df_test = df[df['year'] == year].copy()
        df_test = apply_feature_weights_per_row(df_test, feature_importances)

        X_test = df_test.drop(columns=['FIPS', 'year', 'mortality_rate', 'county_class'])
        y_test = df_test['mortality_rate']
        X_test_scaled = scaler.transform(X_test)

        y_pred = train_mlp(X_train_scaled, y_train, X_test_scaled, y_test, input_dim)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'train_years': f"{min(train_years)}-{max(train_years)}",
            'test_year': year,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })

        fold_results = pd.DataFrame({
            'FIPS': df_test['FIPS'].values,
            'year': df_test['year'].values,
            'True': y_test.values,
            'Predicted': y_pred,
            'Absolute_Error': np.abs(y_test.values - y_pred)
        })
        all_results.append(fold_results)


    results_df = pd.DataFrame(results)
    all_fold_results_df = pd.concat(all_results, ignore_index=True)

    results_df.to_csv(os.path.join(RESULTS_DIR, 'PyTorch_MLP_model_results.csv'), index=False)
    all_fold_results_df.to_csv(os.path.join(RESULTS_DIR, 'PyTorch_MLP_model_preds.csv'), index=False)
    print(f'✅ Saved PyTorch MLP results to {RESULTS_DIR}/PyTorch_MLP_model_results.csv')
    print(f'✅ Saved PyTorch MLP predictions to {RESULTS_DIR}/PyTorch_MLP_model_preds.csv')


RESULTS_CSV = 'County Classification/MLP_Results/PyTorch_MLP_model_results.csv'
OUT_DIR = 'County Classification/MLP_Results/Plots'
os.makedirs(OUT_DIR, exist_ok=True)

def plot_yearly_mae_bar_chart():
    df = pd.read_csv(RESULTS_CSV)
    grouped = df.groupby(['county_class', 'year'])['MAE'].mean().reset_index()

    county_classes = sorted(df['county_class'].unique())

    for county_class in county_classes:
        df_class = grouped[grouped['county_class'] == county_class]

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_class, x='year', y='MAE', palette='viridis')
        plt.xlabel('Year')
        plt.ylabel('Mean Absolute Error')
        plt.title(f'PyTorch MLP Mean Absolute Error – County Class {county_class}')
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f'MAE_BarChart_CountyClass_{county_class}.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {out_path}")

    print("✅ All plots saved!")

def plot_error_histograms():
    df = pd.read_csv(RESULTS_CSV)

    county_classes = sorted(df['county_class'].unique())

    for county_class in county_classes:
        df_class = df[df['county_class'] == county_class]

        years = sorted(df_class['year'].unique())
        for year in years:
            df_year = df_class[df_class['year'] == year]

            # Create histogram of absolute errors
            plt.figure(figsize=(10, 6))
            sns.histplot(df_year['MAE'], bins=20, kde=False, color='skyblue', edgecolor='black')
            plt.xlabel('Absolute Error')
            plt.ylabel('Number of Counties')
            plt.title(f'Error Distribution – County Class {county_class}, Year {year}')
            plt.tight_layout()

            out_path = os.path.join(OUT_DIR, f'Error_Histogram_CountyClass_{county_class}_Year_{year}.png')
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"✅ Saved: {out_path}")

    print("✅ All error histograms saved!")


# def plot_error_histograms_by_year():
#     df = pd.read_csv(RESULTS_CSV)

#     years = sorted(df['year'].unique())

#     for year in years:
#         df_year = df[df['year'] == year]

#         # Combine all folds’ absolute errors into one list
#         all_errors = []
#         for fold_idx in df_year['fold'].unique():
#             fold_data = df_year[df_year['fold'] == fold_idx]
#             all_errors.extend(fold_data['MAE'].values)

#         plt.figure(figsize=(10, 6))
#         sns.histplot(all_errors, bins=20, kde=False, color='skyblue', edgecolor='black')
#         plt.xlabel('Absolute Error')
#         plt.ylabel('Number of Counties')
#         plt.title(f'Error Distribution – Year {year}')
#         plt.tight_layout()

#         out_path = os.path.join(OUT_DIR, f'Error_Histogram_Year_{year}.png')
#         plt.savefig(out_path, dpi=300)
#         plt.close()
#         print(f"✅ Saved: {out_path}")

#     print("✅ All yearly error histograms saved!")

def plot_error_histograms_by_year():
    """
    This function corresponds to the function run_mlp_pipeline_yearly() and plots the error histograms for each year.
    """
    df = pd.read_csv('County Classification/MLP_Results/PyTorch_MLP_model_fold_preds.csv')
    years = sorted(df['year'].unique())

    for year in years:
        df_year = df[df['year'] == year]

        plt.figure(figsize=(10, 6))
        sns.histplot(df_year['Absolute_Error'], bins=20, kde=False, color='skyblue', edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Number of Counties')
        plt.title(f'Error Distribution – Year {year}')
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f'Unified_MLP_Error_Histogram_Year_{year}.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {out_path}")

    print("✅ All yearly error histograms saved!")

def plot_error_histograms_multiyear_train():
    """
    This function corresponds to the function run_mlp_pipeline_multiyear_train() and plots the error histograms for each test year.
    
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    RESULTS_CSV = 'County Classification/MLP_Results/PyTorch_MLP_model_preds.csv'
    OUT_DIR = 'County Classification/MLP_Results/Plots'
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RESULTS_CSV)
    years = sorted(df['year'].unique())

    for year in years:
        df_year = df[df['year'] == year]

        plt.figure(figsize=(10, 6))
        sns.histplot(df_year['Absolute_Error'], bins=20, kde=False, color='skyblue', edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Number of Counties')
        plt.title(f'Unified MLP Error Distribution – Year {year}')
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f'Unified_MLP_Error_Histogram_Year_{year}.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {out_path}")

    print("✅ All yearly error histograms saved!")


if __name__ == "__main__":
    run_mlp_pipeline_multiyear_train()
    plot_error_histograms_multiyear_train()






























# import os
# import pandas as pd
# import numpy as np
# from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import StandardScaler

# # DATA = ['Mortality',
# #         'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
# #         'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
# #         'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
# #         'Single-Parent Household', 'Unemployment']

# FEATURE_IMPORTANCE_CSV = 'County Classification\RF_Mort_Preds_by_Urbanicity\RF_Feat_Imp\RF_feature_importances.csv'
# RESULTS_DIR = 'County Classification/MLP_Results'
# os.makedirs(RESULTS_DIR, exist_ok=True)

# # def prepare_yearly_prediction_data():
# #     # (Use your existing function body here unchanged)
# #     svi_variables = [v for v in DATA if v != 'Mortality']
# #     years = list(range(2010, 2022))
# #     nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
# #     nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
# #     nchs_df = nchs_df.set_index('FIPS')
# #     nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)
# #     mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
# #     mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
# #     mort_df = mort_df.set_index('FIPS')
# #     svi_data = []
# #     for var in svi_variables:
# #         var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
# #         var_df = pd.read_csv(var_path, dtype={'FIPS': str})
# #         var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
# #         long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)
# #         long_df['year'] = long_df['year_var'].str.extract(r'(\\d{4})').astype(int)
# #         long_df = long_df[long_df['year'].between(2010, 2021)]
# #         long_df = long_df.drop(columns='year_var')
# #         svi_data.append(long_df)
# #     from functools import reduce
# #     svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)
# #     for y in years:
# #         mort_col = f'{y+1} MR'
# #         if mort_col not in mort_df.columns:
# #             continue
# #         svi_merged.loc[svi_merged['year'] == y, 'mortality_rate'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(mort_df[mort_col])
# #     svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')
# #     svi_merged = svi_merged.dropna()
# #     return svi_merged

# DATA = ['Mortality',
#         'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
#         'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
#         'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
#         'Single-Parent Household', 'Unemployment']

# def prepare_yearly_prediction_data():
#     """Creates a long-format dataset for predicting next-year mortality using current-year SVI + county class."""
#     svi_variables = [v for v in DATA if v != 'Mortality']
#     years = list(range(2010, 2022))  # We predict mortality up to 2022

#     # Load county category (static)
#     nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
#     nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
#     nchs_df = nchs_df.set_index('FIPS')
#     nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)
    
#     # Load mortality
#     mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
#     mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
#     mort_df = mort_df.set_index('FIPS')
    
#     # Load all SVI variables and reshape to long format per county-year
#     svi_data = []
#     for var in svi_variables:
#         var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
#         var_df = pd.read_csv(var_path, dtype={'FIPS': str})
#         var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
#         long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)

#         # Extract year and filter relevant years
#         long_df['year'] = long_df['year_var'].str.extract(r'(\d{4})').astype(int)
#         long_df = long_df[long_df['year'].between(2010, 2021)]  # we predict 1 year ahead
#         long_df = long_df.drop(columns='year_var')
#         svi_data.append(long_df)

#     # Merge all SVI variables on FIPS + year
#     from functools import reduce
#     svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

#     # Add mortality for year+1
#     for y in years:
#         mort_col = f'{y+1} MR'
#         if mort_col not in mort_df.columns:
#             continue
#         svi_merged.loc[svi_merged['year'] == y, 'mortality_rate'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(mort_df[mort_col])

#     # Add county class
#     svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

#     # Drop rows with any missing values
#     svi_merged = svi_merged.dropna()

#     return svi_merged

# def load_feature_importances():
#     fi_df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
#     # Group by county_class and feature
#     grouped = fi_df.groupby(['county_class', 'Feature'])['Importance'].mean().reset_index()
#     # Normalize per county_class
#     grouped['Importance'] = grouped.groupby('county_class')['Importance'].transform(lambda x: x / x.sum())
#     return grouped

# def apply_feature_weights(df, feature_weights, county_class):
#     df_copy = df.copy()
#     features = [f for f in df.columns if f in DATA and f != 'Mortality']
#     for feature in features:
#         weight = feature_weights.loc[
#             (feature_weights['county_class'] == county_class) & 
#             (feature_weights['Feature'] == feature),
#             'Importance'
#         ]
#         if not weight.empty:
#             df_copy[feature] = df_copy[feature] * weight.values[0]
#     return df_copy

# def run_mlp_pipeline():
#     df = prepare_yearly_prediction_data()
#     feature_importances = load_feature_importances()

#     results = []
#     county_classes = sorted(df['county_class'].unique())

#     for county_class in county_classes:
#         print(f'🚀 Training MLP for County Class: {county_class}')
#         df_sub = df[df['county_class'] == county_class].copy()

#         for year in range(2010, 2022):
#             df_year = df_sub[df_sub['year'] == year].copy()
#             if df_year.empty:
#                 continue

#             df_year = apply_feature_weights(df_year, feature_importances, county_class)
#             X = df_year.drop(columns=['FIPS', 'year', 'mortality_rate', 'county_class'])
#             y = df_year['mortality_rate']

#             # Scale features
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X)

#             kf = KFold(n_splits=5, shuffle=True, random_state=42)
#             for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
#                 X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
#                 y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#                 model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)

#                 rmse = mean_squared_error(y_test, y_pred, squared=False)
#                 mae = mean_absolute_error(y_test, y_pred)
#                 r2 = r2_score(y_test, y_pred)

#                 results.append({
#                     'county_class': county_class,
#                     'year': year,
#                     'fold': fold_idx + 1,
#                     'RMSE': rmse,
#                     'MAE': mae,
#                     'R2': r2
#                 })

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(os.path.join(RESULTS_DIR, 'MLP_model_results.csv'), index=False)
#     print(f'✅ Saved MLP results to {RESULTS_DIR}/MLP_model_results.csv')

# if __name__ == "__main__":
#     run_mlp_pipeline()

