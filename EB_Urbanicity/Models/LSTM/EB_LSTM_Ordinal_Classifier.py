### 8/2/25, EB: Ok, so the straightforward classification with an LSTM approach is not really working. Even binning the zero-mortality counties into a separate class isn't helping much. It made a difference for sure,
### but the results are still terrible for many of the classes. I like the LSTM approach, though, so what I'm going to try now is to use the spacecutter package to wrap the LSTM output in an ordinal regression model.
### This should allow us to leverage the LSTM's prediction capabilities while now using the ordinal regression model to handle the classification task.


from spacecutter.models import OrdinalLogisticModel
from spacecutter.losses import cumulative_link_loss

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


### 8/2/25, EB: ChatGPT tried to import a function called `ordinal_logits_to_class` that spacecutter doesn't actually have, so I wrote my own.
### Similarly, it tried to import a class called `OrdinalLogisticLoss` that spacecutter doesn't have, so I wrote my own.


# class OrdinalLogisticLoss(nn.Module):
#     def __init__(self, reduction='mean'):
#         super().__init__()
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         return cumulative_link_loss(logits, targets, reduction=self.reduction)


# def ordinal_logits_to_class(logits, threshold=0.5):
#     """
#     Converts ordinal logits to predicted class index.
    
#     Args:
#         logits: Tensor of shape (batch_size, K-1) from OrdinalLogisticModel
#         threshold: Classification threshold (typically 0.5)
    
#     Returns:
#         Tensor of shape (batch_size,) with predicted class indices (0 to K-1)
#     """
#     prob = torch.sigmoid(logits)
#     return torch.sum(prob > threshold, dim=1)
def ordinal_logits_to_class(logits: torch.Tensor) -> torch.Tensor:
    """
    Given logits from an ordinal model, return predicted class indices.

    The class with the highest predicted probability is selected.
    Assumes logits have shape (batch_size, num_classes).
    """
    return torch.argmax(logits, dim=1)


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

# def prepare_lstm_dataset_with_tracking(df, svi_variables, sequence_length=3):
#     """
#     Prepares LSTM-ready sequences for mortality prediction.
#     Each X is a (sequence_length, n_features) array of SVI data,
#     with an optional static feature vector (urbanicity one-hot).
#     Returns sequences, static features, targets, FIPS, and years.
#     """

#     # Encode county_class (urbanicity) as one-hot
#     enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
#     county_class_encoded = enc.fit_transform(df[['county_class']])
#     df_encoded = df.copy()
#     for i, col in enumerate(enc.get_feature_names_out(['county_class'])):
#         df_encoded[col] = county_class_encoded[:, i]

#     # Normalize SVI variables individually
#     scalers = {var: StandardScaler() for var in svi_variables}
#     for var in svi_variables:
#         df_encoded[var] = scalers[var].fit_transform(df_encoded[[var]])

#     sequences = []
#     static_features = []
#     targets = []
#     fips_list = []
#     year_list = []

#     grouped = df_encoded.groupby('FIPS')

#     for fips, group in grouped:
#         group = group.sort_values('year')
#         if len(group) < sequence_length + 1:
#             continue

#         # Verify urbanicity is consistent
#         urban_vals = group[enc.get_feature_names_out(['county_class'])].drop_duplicates()
#         if len(urban_vals) != 1:
#             print(f"⚠️ Urbanicity mismatch for FIPS {fips}, skipping.")
#             continue
#         urban_static = urban_vals.values[0]

#         svi_seq = group[svi_variables].values
#         mortality_seq = group['mortality_next'].values
#         years = group['year'].values

#         for i in range(len(group) - sequence_length):
#             x_seq = svi_seq[i:i+sequence_length]
#             y_target = mortality_seq[i + sequence_length]
#             target_year = years[i + sequence_length]

#             if np.isnan(y_target):
#                 continue

#             sequences.append(x_seq)
#             static_features.append(urban_static)
#             targets.append(y_target)
#             fips_list.append(fips)
#             year_list.append(target_year)

#     print(f"✅ Built {len(sequences)} valid sequences.")
#     return (
#         np.array(sequences),
#         np.array(static_features),
#         np.array(targets),
#         np.array(fips_list),
#         np.array(year_list)
#     )

def prepare_lstm_dataset_with_tracking_classification(df, svi_variables, sequence_length=3):
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Encode urbanicity as one-hot
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    county_class_encoded = enc.fit_transform(df[['county_class']])
    df_encoded = df.copy()
    for i, col in enumerate(enc.get_feature_names_out(['county_class'])):
        df_encoded[col] = county_class_encoded[:, i]

    # Normalize SVI variables
    scalers = {var: StandardScaler() for var in svi_variables}
    for var in svi_variables:
        df_encoded[var] = scalers[var].fit_transform(df_encoded[[var]])

    # Bin mortality rates into deciles per year
    df_encoded['mortality_decile'] = df_encoded.groupby('year')['mortality_next'] \
        .rank(pct=True).apply(lambda x: min(int(x * 10), 9))

    # Build sequences
    sequences, static_features, targets, fips_list, year_list = [], [], [], [], []
    grouped = df_encoded.groupby('FIPS')

    for fips, group in grouped:
        group = group.sort_values('year')
        if len(group) < sequence_length + 1:
            continue

        # Ensure static urbanicity
        urban_vals = group[enc.get_feature_names_out(['county_class'])].drop_duplicates()
        if len(urban_vals) != 1:
            print(f"⚠️ Urbanicity mismatch for FIPS {fips}, skipping.")
            continue
        urban_static = urban_vals.values[0]

        svi_seq = group[svi_variables].values
        class_seq = group['mortality_decile'].values
        years = group['year'].values

        for i in range(len(group) - sequence_length):
            x_seq = svi_seq[i:i+sequence_length]
            y_target = class_seq[i + sequence_length]
            target_year = years[i + sequence_length]

            if np.isnan(y_target):
                continue

            sequences.append(x_seq)
            static_features.append(urban_static)
            targets.append(int(y_target))
            fips_list.append(fips)
            year_list.append(target_year)

    print(f"✅ Built {len(sequences)} valid sequences for classification.")
    return (
        np.array(sequences),
        np.array(static_features),
        np.array(targets),
        np.array(fips_list),
        np.array(year_list)
    )

######
def preprocess_with_zero_class(df, svi_variables):
    """
    Adds an 11-class target where class 0 represents zero mortality,
    and classes 1-10 are deciles of non-zero mortality.
    """
    df = df.copy()
    df['mortality_decile'] = np.nan

    for year, group in df.groupby('year'):
        nonzero_mask = group['mortality_next'] > 0
        zero_mask = ~nonzero_mask

        df.loc[group.index[zero_mask], 'mortality_decile'] = 0

        nonzero = group[nonzero_mask].copy()
        if len(nonzero) == 0:
            continue

        ranked = nonzero['mortality_next'].rank(pct=True)
        binned = ranked.apply(lambda x: min(int(x * 10), 9) + 1)  # shift by 1
        df.loc[nonzero.index, 'mortality_decile'] = binned.astype(int)

    return df

def build_lstm_classification_sequences(df, svi_variables, sequence_length=3):
    # Encode county_class as one-hot
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    county_class_encoded = enc.fit_transform(df[['county_class']])
    df_encoded = df.copy()
    for i, col in enumerate(enc.get_feature_names_out(['county_class'])):
        df_encoded[col] = county_class_encoded[:, i]

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

        urban_vals = group[enc.get_feature_names_out(['county_class'])].drop_duplicates()
        if len(urban_vals) != 1:
            continue
        urban_static = urban_vals.values[0]

        svi_seq = group[svi_variables].values
        class_seq = group['mortality_decile'].values
        years = group['year'].values

        for i in range(len(group) - sequence_length):
            x_seq = svi_seq[i:i+sequence_length]
            y_target = class_seq[i + sequence_length]
            target_year = years[i + sequence_length]

            if np.isnan(y_target):
                continue

            sequences.append(x_seq)
            static_features.append(urban_static)
            targets.append(int(y_target))
            fips_list.append(fips)
            year_list.append(target_year)

    return (
        np.array(sequences),
        np.array(static_features),
        np.array(targets),
        np.array(fips_list),
        np.array(year_list)
    )

#####

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

# class MortalityDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, static_features, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.static_features[idx], self.targets[idx]


# class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, static_size, num_layers=1, dropout=0.0, num_classes=11):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size + static_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, num_classes)  # logits for classification
        # )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        last_output = lstm_out[:, -1, :]
        combined = torch.cat((last_output, x_static), dim=1)
        return self.fc(combined)  # shape: (batch_size, num_classes)

class MortalityDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, static_features, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)  # changed to long!

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.static_features[idx], self.targets[idx]


# class LSTMOrdinalClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, static_size, num_layers=1, dropout=0.0, num_classes=11):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size, hidden_size, num_layers, batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )

#         self.shared_fc = nn.Sequential(
#             nn.Linear(hidden_size + static_size, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 1)  # Output a single scalar per sample
#         )

#         # The ordinal logistic wrapper: models cutpoints for K ordered classes
#         self.ordinal_model = OrdinalLogisticModel(num_features=1, n_classes=num_classes)

#     def forward(self, x_seq, x_static):
#         lstm_out, _ = self.lstm(x_seq)
#         last_output = lstm_out[:, -1, :]
#         combined = torch.cat((last_output, x_static), dim=1)

#         scalar_output = self.shared_fc(combined)  # shape: (batch_size, 1)
#         #output = self.ordinal_model(scalar_output)  # shape: (batch_size, num_classes - 1)

#         return scalar_output

class LSTMOrdinalClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, static_size, num_layers=1, dropout=0.0, num_classes=11):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Shared fully connected head that outputs a single scalar per sample
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size + static_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Output shape: (batch_size, 1)
        )

        # Wrap the scalar-output FC head in an OrdinalLogisticModel
        self.ordinal_model = OrdinalLogisticModel(
            predictor=self.shared_fc,         # FC module that outputs [B, 1]
            num_classes=num_classes,          # e.g., 11 for deciles + zeros
            init_cutpoints='ordered'          # or 'random'
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        last_output = lstm_out[:, -1, :]                   # shape: (batch_size, hidden_size)
        combined = torch.cat((last_output, x_static), dim=1)  # shape: (batch_size, hidden + static)

        return self.ordinal_model(combined)  # shape: (batch_size, num_classes)


# def train_classifier(y_train, model, train_loader, val_loader, n_epochs=30, lr=1e-3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     from sklearn.utils.class_weight import compute_class_weight

#     # Get unique classes from training targets
#     classes = np.unique(y_train)
#     weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
#     class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

#     # Use in loss
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     #criterion = nn.CrossEntropyLoss()
#     model = model.to(device)

#     best_val_loss = float('inf')
#     best_model_state = None

#     for epoch in range(n_epochs):
#         model.train()
#         total_loss = 0
#         for x_seq, x_static, y in train_loader:
#             x_seq, x_static = x_seq.to(device), x_static.to(device)
#             y = y.to(device).long()  # for CrossEntropyLoss

#             optimizer.zero_grad()
#             logits = model(x_seq, x_static)
#             loss = criterion(logits, y)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             total_loss += loss.item()

#         # Validation
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for x_seq, x_static, y in val_loader:
#                 x_seq, x_static = x_seq.to(device), x_static.to(device)
#                 y = y.to(device).long()
#                 logits = model(x_seq, x_static)
#                 val_loss += criterion(logits, y).item()

#         avg_train_loss = total_loss / len(train_loader)
#         avg_val_loss = val_loss / len(val_loader)
#         if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
#             print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_state = model.state_dict()

#     if best_model_state:
#         model.load_state_dict(best_model_state)

#     return model


# def train_classifier_ordinal(y_train, model, train_loader, val_loader, n_epochs=30, lr=1e-3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ordinal_model = OrdinalLogisticModel(n_classes=11).to(device)  # Same number of classes as before

    
#     # Use ordinal logistic loss (no need for class weights)
#     criterion = OrdinalLogisticLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     model = model.to(device)

#     best_val_loss = float('inf')
#     best_model_state = None

#     for epoch in range(n_epochs):
#         model.train()
#         total_loss = 0
#         for x_seq, x_static, y in train_loader:
#             x_seq, x_static = x_seq.to(device), x_static.to(device)
#             y = y.to(device).long()  # target is integer class index (0 to K-1)

#             optimizer.zero_grad()
#             #logits = model(x_seq, x_static)  # shape: (batch_size, K-1)
#             theta = model(x_seq, x_static)              # (batch_size, 1)
#             logits = ordinal_model(theta)               # (batch_size, n_classes - 1)
#             loss = criterion(logits, y)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             total_loss += loss.item()

#         # Validation
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for x_seq, x_static, y in val_loader:
#                 x_seq, x_static = x_seq.to(device), x_static.to(device)
#                 y = y.to(device).long()
#                 #logits = model(x_seq, x_static)
#                 theta = model(x_seq, x_static)
#                 logits = ordinal_model(theta)
#                 val_loss += criterion(logits, y).item()

#         avg_train_loss = total_loss / len(train_loader)
#         avg_val_loss = val_loss / len(val_loader)
#         if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
#             print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_state = model.state_dict()

#     if best_model_state:
#         model.load_state_dict(best_model_state)

#     return model

##################
### 8/3/25, EB: Even the ordinal model isn't performing great. I'm going to try to manually implement class weightings in the loss
### function, since the spacecutter implementation doesn't seem to support it directly. Hopefully this will help the model learn the
### higher mortality classes better, since they are so heavily underrepresented in the data.

from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(y_train, n_classes):
    weights = compute_class_weight(class_weight='balanced',
                                   classes=np.arange(n_classes),
                                   y=y_train)
    return torch.tensor(weights, dtype=torch.float32)

# class WeightedOrdinalLoss(nn.Module):
#     def __init__(self, class_weights):
#         super().__init__()
#         self.class_weights = class_weights  # tensor of shape (n_classes,)
    
#     def forward(self, logits, targets):
#         # logits: (batch_size, n_classes - 1)
#         # targets: (batch_size,)
#         targets = targets.view(-1)  # ensure 1D
#         unweighted_loss = cumulative_link_loss(logits, targets, reduction='none')  # (batch_size,)
#         sample_weights = self.class_weights[targets]  # (batch_size,)
#         weighted_loss = unweighted_loss * sample_weights
#         return weighted_loss.mean()

class WeightedOrdinalLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights  # shape: (num_classes,)

    def forward(self, logits, targets):
        # FIX: targets must be 2D for gather()
        targets = targets.unsqueeze(1)  # shape: (batch_size, 1)

        # Compute unweighted loss with 'none' reduction
        unweighted_loss = cumulative_link_loss(logits, targets, reduction='none')  # shape: (batch_size,)

        # Apply per-sample weight based on target class
        sample_weights = self.class_weights[targets.squeeze()]  # shape: (batch_size,)
        weighted_loss = unweighted_loss * sample_weights

        return weighted_loss.mean()

##################



def train_classifier_ordinal(y_train, model, train_loader, val_loader, n_epochs=30, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### 8/3/25, EB: Using the weighted class loss function defined directly above.
    # Use ordinal logistic loss directly
    #criterion = lambda logits, targets: cumulative_link_loss(logits, targets, reduction='elementwise_mean')
    
    class_weights = get_class_weights(y_train, n_classes=11).to(device)
    criterion = WeightedOrdinalLoss(class_weights)
    ####
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x_seq, x_static, y in train_loader:
            x_seq, x_static = x_seq.to(device), x_static.to(device)
            y = y.to(device).long()  # Target class index (0 to num_classes - 1)

            optimizer.zero_grad()
            logits = model(x_seq, x_static)   # Already outputs shape: (batch_size, num_classes)
            loss = criterion(logits, y)#.unsqueeze(1))  # 8/3/25, EB: Ensure targets are 2D for cumulative_link_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_seq, x_static, y in val_loader:
                x_seq, x_static = x_seq.to(device), x_static.to(device)
                y = y.to(device).long()
                logits = model(x_seq, x_static)
                val_loss += criterion(logits, y).item()#.unsqueeze(1)).item() # 8/3/25, EB: Ensure targets are 2D for cumulative_link_loss

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model



# def get_predictions(model, val_loader):
#     model.eval()
#     device = next(model.parameters()).device
#     y_true = []
#     y_pred = []

#     with torch.no_grad():
#         for x_seq, x_static, y in val_loader:
#             x_seq, x_static = x_seq.to(device), x_static.to(device)
#             preds = model(x_seq, x_static).cpu().numpy()
#             y_pred.extend(preds)
#             y_true.extend(y.numpy())

#     return np.array(y_true), np.array(y_pred)

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
#         residuals = np.abs(df_year['Pred_MR'] - df_year['True_MR'])
#         preds = df_year['Pred_MR']

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

# def get_predictions(model, val_loader):
#     model.eval()
#     device = next(model.parameters()).device
#     y_true = []
#     y_pred = []

#     with torch.no_grad():
#         for x_seq, x_static, y in val_loader:
#             x_seq, x_static = x_seq.to(device), x_static.to(device)
#             y = y.to(device).long()

#             logits = model(x_seq, x_static)
#             preds = torch.argmax(logits, dim=1)

#             y_pred.extend(preds.cpu().numpy())
#             y_true.extend(y.cpu().numpy())

#     return np.array(y_true), np.array(y_pred)


# def get_predictions(model, val_loader):
#     model.eval()
#     device = next(model.parameters()).device
#     y_true = []
#     y_pred = []
    
#     # Use the same ordinal transformation used during training
#     ordinal_model = OrdinalLogisticModel(n_classes=11).to(device)

#     with torch.no_grad():
#         for x_seq, x_static, y in val_loader:
#             x_seq, x_static = x_seq.to(device), x_static.to(device)
#             y = y.to(device).long()

#             #logits = model(x_seq, x_static)  # shape: (batch_size, K-1)
#             theta = model(x_seq, x_static)             # shape: (batch_size, 1)
#             logits = ordinal_model(theta)              # shape: (batch_size, 10)
#             preds = ordinal_logits_to_class(logits)  # returns class indices

#             y_pred.extend(preds.cpu().numpy())
#             y_true.extend(y.cpu().numpy())

#     return np.array(y_true), np.array(y_pred)

def get_predictions(model, val_loader):
    model.eval()
    device = next(model.parameters()).device
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_seq, x_static, y in val_loader:
            x_seq, x_static = x_seq.to(device), x_static.to(device)
            y = y.to(device).long()

            logits = model(x_seq, x_static)  # Already returns shape: (batch_size, num_classes)
            preds = ordinal_logits_to_class(logits)  # Map logits to predicted class indices

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


def plot_classification_summary_by_year(results_df, out_dir=None):
    """
    For each year in the results:
      - Left: Count of correct vs incorrect predictions.
      - Right: Histogram of predicted class frequencies.

    If `out_dir` is given, saves plots instead of displaying.
    """
    years = sorted(results_df['Year'].unique())

    for year in years:
        df_year = results_df[results_df['Year'] == year].copy()
        df_year['Correct'] = df_year['Pred_Class'] == df_year['True_Class']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Correct vs Incorrect
        sns.countplot(data=df_year, x='Correct', ax=axes[0], palette='pastel')
        axes[0].set_title(f"{year} Prediction Accuracy")
        axes[0].set_xlabel("Prediction Correct?")
        axes[0].set_ylabel("Count")
        axes[0].grid(True)

        # Predicted Class Distribution
        sns.histplot(df_year['Pred_Class'], bins=np.arange(11)-0.5, ax=axes[1], color='salmon', edgecolor='black')
        axes[1].set_title(f"{year} Predicted Class Distribution")
        axes[1].set_xlabel("Predicted Class")
        axes[1].set_ylabel("Frequency")
        axes[1].set_xticks(range(10))
        axes[1].grid(True)

        plt.tight_layout()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/classification_summary_{year}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()



def plot_avg_abs_error_by_class(y_true, y_pred, title="Average Absolute Error by True Class"):
    """
    Plots the average absolute prediction error for each true class label.
    
    Parameters:
        y_true (array-like): Ground truth class labels (integers from 0 to K-1).
        y_pred (array-like): Predicted class labels (same format as y_true).
        title (str): Title of the plot.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    error_df = pd.DataFrame({
        "true": y_true,
        "error": np.abs(y_true - y_pred)
    })

    avg_errors = error_df.groupby("true")["error"].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(avg_errors.index, avg_errors.values, color='steelblue')
    plt.xlabel("True Class Label")
    plt.ylabel("Average Absolute Error")
    plt.title(title)
    plt.xticks(avg_errors.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


### 8/2/25, EB: This first main function is the original one, that uses 10 classes directly from the empirical distribution.
### The second main function below is the one that uses the 11-class approach with a zero class for zero mortality counties.
# def main():
#     ### 6/18/25, EB: Results are below, I found that the best architecture was a 64 hidden size, 2 layers, and a learning rate of 0.0005.
#     ### However, a single hidden layer only increased the overall RMSE by 0.1, so we'll go with that for now.
#     set_random_seeds()
#     sequence_length = 1  # Number of previous years to use for prediction
    
#     print('Running LSTM model for mortality prediction...')
#     # Prepare the dataset
#     df = prepare_yearly_prediction_data_mortality()
#     X_seq, X_static, y, fips_arr, year_arr = prepare_lstm_dataset_with_tracking_classification(df, svi_variables=DATA[1:], sequence_length=sequence_length)
#     print(year_arr)

#     # Want to do temporal predictions, so split the data by year
#     data_set = yearly_data_split(X_seq, X_static, y, fips_arr, year_arr, year=2020)
#     X_train_seq, X_val_seq, X_train_static, X_val_static, y_train, y_val, fips_train, fips_val, year_train, year_val = data_set



#     # Sanity check on what years I'm using to train and validate
#     print("Training years:", np.unique(year_train))
#     print("Validation years:", np.unique(year_val))
#     print(f"Using {sequence_length} previous years for prediction.")
#     # Dataloaders
#     train_ds = MortalityDataset(X_train_seq, X_train_static, y_train)
#     val_ds = MortalityDataset(X_val_seq, X_val_static, y_val)
#     train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=64)

#     # Model
#     model = LSTMClassifier(input_size=X_seq.shape[2], hidden_size=64, static_size=X_static.shape[1], dropout=0.2)
#     trained_model = train_classifier(y_train, model, train_loader, val_loader, n_epochs=150, lr=5e-3)
#     print('Model training complete.')

#     # Get preditions
#     y_true, y_pred = get_predictions(trained_model, val_loader)
#     from sklearn.metrics import classification_report
#     print(classification_report(y_true, y_pred, digits=3))

#     results_df = pd.DataFrame({
#         'FIPS': fips_val,
#         'Year': year_val,
#         'True_Class': y_true,
#         'Pred_Class': y_pred
#     })

#     # os.makedirs('County Classification\LSTM_MR_Preds', exist_ok=True)
#     # results_df.to_csv('County Classification\LSTM_MR_Preds\LSTM_MR_predictions_threeyear.csv', index=False)
#     # print("✅ Saved LSTM predictions to County Classification\LSTM_MR_Preds\LSTM_predictions_threeyear.csv")
#     plot_classification_summary_by_year(results_df)#, out_dir='County Classification/LSTM_MR_Plots')

def main():

    set_random_seeds()
    sequence_length = 1  # Number of previous years to use for prediction
    
    print('Running LSTM model for mortality prediction...')
    # Prepare the dataset
    df = prepare_yearly_prediction_data_mortality()
    # Step 2: Preprocess to assign 11-class targets (0 = zero-mortality, 1–10 = deciles)
    df_processed = preprocess_with_zero_class(df, svi_variables=DATA[1:])

    # Step 3: Build LSTM sequences for classification
    X_seq, X_static, y, fips_arr, year_arr = build_lstm_classification_sequences(
        df_processed, svi_variables=DATA[1:], sequence_length=sequence_length
    )
    
    # To maximize the model's perforamnce, we train up to year 2020 (which is really the year 2021), and validate on 2022.
    data_set = yearly_data_split(X_seq, X_static, y, fips_arr, year_arr, year=2020)
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

    # Model
    model = LSTMOrdinalClassifier(input_size=X_seq.shape[2], hidden_size=64, static_size=X_static.shape[1], dropout=0.2)
    trained_model = train_classifier_ordinal(y_train, model, train_loader, val_loader, n_epochs=150, lr=5e-3)
    print('Model training complete.')

    # Get preditions
    y_true, y_pred = get_predictions(trained_model, val_loader)
    from sklearn.metrics import classification_report, confusion_matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print('-------------------------------------------')
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    

    results_df = pd.DataFrame({
        'FIPS': fips_val,
        'Year': year_val,
        'True_Class': y_true,
        'Pred_Class': y_pred
    })

    # os.makedirs('County Classification\LSTM_MR_Preds', exist_ok=True)
    # results_df.to_csv('County Classification\LSTM_MR_Preds\LSTM_MR_predictions_threeyear.csv', index=False)
    # print("✅ Saved LSTM predictions to County Classification\LSTM_MR_Preds\LSTM_predictions_threeyear.csv")
    plot_classification_summary_by_year(results_df)#, out_dir='County Classification/LSTM_MR_Plots')
    plot_avg_abs_error_by_class(y_true, y_pred, title="Average Absolute Error by True Class (with Zero Class)")
    

if __name__ == "__main__":
    main() 