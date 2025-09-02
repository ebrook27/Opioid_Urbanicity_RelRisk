### 6/25/25, EB: Talked with Andrew the other day, and we tossed around the idea of using a MLP to predict urbanicity.
### If we could use the SVI data to predict urbanicity, that would diversify our analysis, and demonstrate that urbanicity
### is a key component to our study. Not only does urbanicity inform the mortality, but SVI informs the urbanicity.
### My first attempt here will be to use an MLP to predict urbanicity from the SVI data. If this works, we can complicate things more from there.

import pandas as pd
from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import classification_report

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

import pandas as pd
from functools import reduce

def prepare_yearly_prediction_data_mortality_unstaggered():
    """
    Prepares a long-format dataset for predicting county class or other outcomes
    using current-year SVI + same-year Mortality Rate.
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

    # Add same-year Mortality Rate (not staggered)
    for y in years:
        mr_col = f'{y} MR'
        if mr_col not in mort_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'Mortality'] = svi_merged.loc[
            svi_merged['year'] == y, 'FIPS'].map(mort_df[mr_col])

    # Add urbanicity class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows with missing data
    svi_merged = svi_merged.dropna()

    return svi_merged


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), num_classes=6, dropout=0.2):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes)
        )

    def forward(self, x):
        return self.model(x)

def tensorize(df):
    """
    Converts a DataFrame to PyTorch tensors.
    """
    # Features and target
    # Testing including mortality as an input feature
    # X = df[[v for v in DATA if v != 'Mortality']].values
    X = df[[v for v in DATA]].values
    # Testing only 3 urbanicity classes for simplicity
    #y = df['county_class'].astype(str).values  # Make sure it's string labels
    y = df['urban_class_grouped'].values
   
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    input_dim = X_train.shape[1]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, le, input_dim

def tensorize_for_plotting(df):
    """
    Converts a DataFrame to PyTorch tensors and returns FIPS codes for test samples.
    """
    # Features and target
    X = df[[v for v in DATA]].values
    y = df['urban_class_grouped'].values
    fips = df['FIPS'].values  # track FIPS for visualization

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test split (now includes FIPS)
    X_train, X_test, y_train, y_test, fips_train, fips_test = train_test_split(
        X, y_encoded, fips,
        test_size=0.2, random_state=42, stratify=y_encoded
    )

    input_dim = X_train.shape[1]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

    return (
        X_train_tensor, y_train_tensor,
        X_test_tensor, y_test_tensor,
        le, input_dim,
        fips_test  # new addition
    )


def train_mlp_model(train_loader, test_loader, input_dim, num_classes, target_names, n_epochs=30, lr=1e-3):
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(y_batch.numpy())

    report = classification_report(all_targets, all_preds, target_names=target_names)
    return model, report

def train_mlp_model_for_plotting(train_loader, test_loader, input_dim, num_classes, target_names, 
                    test_fips=None, return_predictions=False, n_epochs=30, lr=1e-3):
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(y_batch.numpy())

    report = classification_report(all_targets, all_preds, target_names=target_names)

    if return_predictions:
        if test_fips is None:
            raise ValueError("Must pass `test_fips` if return_predictions=True")
        result_df = pd.DataFrame({
            'FIPS': test_fips,
            'TrueClass': all_targets,
            'PredClass': all_preds
        })
        result_df['Correct'] = result_df['TrueClass'] == result_df['PredClass']
        return model, report, result_df

    return model, report


### 6/26/25,EB: I'm now trying to visualize the accuracy of the MLP model on a map.
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os

# ── CONFIG ─────────────────────────────────────────────────────────────
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
OUTPUT_DIR = 'Urbanicity_MLP_Maps'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_prediction_accuracy_map(results_df, shapefile_path, out_path):
    # Load shapefile and prep FIPS
    gdf = gpd.read_file(shapefile_path)
    gdf['FIPS'] = gdf['FIPS'].astype(str).str.zfill(5)

    # Merge predictions
    df = gdf.merge(results_df, on='FIPS', how='left')

    # Accuracy column
    df['Correct'] = df['TrueClass'] == df['PredClass']
    # df['Outcome'] = df['Correct'].map({True: 'Correct', False: 'Incorrect'})
    df['Outcome'] = df['Correct'].map({True: 'Correct', False: 'Incorrect'})


    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    df.plot(column='Outcome', ax=ax, categorical=True, legend=True,
            cmap='RdYlGn', edgecolor='black', linewidth=0.1,
            missing_kwds={'color': 'lightgrey', 'label': 'No Data'})

    ax.set_title('MLP Urbanicity Prediction Accuracy (Test Set)', fontsize=14)
    ax.axis('off')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved prediction accuracy map to: {out_path}")

def plot_prediction_accuracy_map_CONUS(results_df, shapefile_path, out_path):
    # Load shapefile and prep FIPS
    gdf = gpd.read_file(shapefile_path)
    gdf['FIPS'] = gdf['FIPS'].astype(str).str.zfill(5)

    # Exclude Alaska (02) and Hawaii (15)
    gdf = gdf[~gdf['STATEFP'].isin(['02', '15'])]

    # Merge predictions
    df = gdf.merge(results_df, on='FIPS', how='left')

    # Accuracy column
    df['Correct'] = df['TrueClass'] == df['PredClass']
    df['Outcome'] = df['Correct'].map({True: 'Correct', False: 'Incorrect'})

    # Set correct category order for coloring
    df['Outcome'] = pd.Categorical(df['Outcome'], categories=['Incorrect', 'Correct'], ordered=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    df.plot(column='Outcome', ax=ax, categorical=True, legend=True,
            cmap='RdYlGn', edgecolor='black', linewidth=0.1,
            missing_kwds={'color': 'lightgrey', 'label': 'No Data'})

    ax.set_title('MLP Urbanicity Prediction Accuracy (Test Set)', fontsize=14)
    ax.axis('off')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved prediction accuracy map to: {out_path}")




def main():
    # Load and prepare data
    df = prepare_yearly_prediction_data_mortality_unstaggered()
    #print(df.head())
    # Map old categories to new ones
    urbanicity_map = {
        '1': 'Urban',
        '2': 'Urban',
        '3': 'Suburban',
        '4': 'Suburban',
        '5': 'Rural',
        '6': 'Rural'
    }

    # Apply to dataframe
    df['urban_class_grouped'] = df['county_class'].astype(str).map(urbanicity_map)

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, le, input_dim, fips_test = tensorize_for_plotting(df)
    num_classes = len(le.classes_)
    target_names = le.classes_
    
    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

    model, report, result_df = train_mlp_model_for_plotting(train_loader, test_loader, input_dim, num_classes, target_names,test_fips=fips_test, return_predictions=True)
    # print(result_df.head())
    plot_prediction_accuracy_map_CONUS(result_df, SHAPE_PATH, os.path.join(OUTPUT_DIR, 'mlp_urbanicity_accuracy_map.png'))
    print(report)
    
if __name__ == "__main__":
    main()