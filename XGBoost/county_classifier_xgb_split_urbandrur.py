### 4/2/25, EB: Before I give up on trying to classify all 6 county categories,
### I want to try manually splitting the data into urban and rural categories, and then train a model on each.
### This way we might be able to salvage the XGB feature ranking. If this doesn't work, I'll either need
### to get an idea from AD, AS, or VM, or try to just use urban vs rural more broadly.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']#, 'Social Capital', 'Social Connectedness']# 'Family Unity', 'Community Health',

# def construct_data_df():
#     """Builds a dataset with aggregated temporal features."""
#     # Load mortality and social capital (unchanged)
#     mortality_path = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
#     data_df = pd.read_csv(mortality_path, dtype={'FIPS': str})
#     data_df['FIPS'] = data_df['FIPS'].str.zfill(5)

#     # Load social capital (static 2018 data)
#     sci_df = pd.read_csv(
#         'Data/SVI/Final Files/Social Capital_final_rates.csv',
#         usecols=['FIPS', '2018 Social Capital'],
#         dtype={'FIPS': str}
#     )
#     sci_df['FIPS'] = sci_df['FIPS'].str.zfill(5)
#     data_df = pd.merge(data_df, sci_df, on='FIPS', how='left')

#     # Process time-varying variables (e.g., unemployment, housing burden)
#     for variable in [v for v in DATA if v not in ['Mortality', 'Social Capital']]:
#         variable_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
#         var_df = pd.read_csv(
#             variable_path,
#             dtype={'FIPS': str},
#             header=0,
#             names=['FIPS'] + [f'{year} {variable}' for year in range(2010, 2023)]
#         )
#         var_df['FIPS'] = var_df['FIPS'].str.zfill(5)

#         # Melt to long format for temporal aggregation
#         var_df_melted = var_df.melt(
#             id_vars='FIPS',
#             var_name='Year',
#             value_name=variable
#         )
#         var_df_melted['Year'] = var_df_melted['Year'].str.extract('(\d+)').astype(int)
        
#         # Compute aggregations
#         # aggregations = var_df_melted.groupby('FIPS').agg({
#         #     variable: [
#         #         ('mean', np.mean),
#         #         ('std', np.std),
#         #         ('slope', lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]),
#         #         ('last', lambda x: x.iloc[-1])
#         #     ]
#         # }).reset_index()
        
#         ### 3/27/25, EB: Got a warning regarding the std function and np.polyfit, so I made the following changes:
#         def safe_slope(x):
#             """Compute slope only if there are ≥2 data points and variance > 0."""
#             if len(x) < 2 or np.var(x) == 0:
#                 return np.nan  # Return NaN for invalid slopes
#             return np.polyfit(np.arange(len(x)), x, 1)[0]

#         # Update the aggregation to use safe_slope
#         aggregations = var_df_melted.groupby('FIPS').agg({
#             variable: [
#                 ('mean', 'mean'),
#                 ('std', 'std'),
#                 ('slope', safe_slope),  # Use the safeguarded function
#                 ('last', 'last')
#             ]
#         }).reset_index()
        
        
#         # Flatten column names
#         aggregations.columns = [
#             'FIPS',
#             f'{variable}_mean',
#             f'{variable}_std',
#             f'{variable}_slope',
#             f'{variable}_last'
#         ]
        
#         # Merge into main dataframe
#         data_df = pd.merge(data_df, aggregations, on='FIPS', how='left')
        
        
#         # # After merging aggregations into data_df:
#         data_df = data_df.dropna(subset=[f'{variable}_slope'])  # Drop rows with invalid slopes
#         # # OR
#         #data_df[f'{variable}_slope'] = data_df[f'{variable}_slope'].fillna(0)  # Fill with 0

#     # Load and merge urban-rural classification
#     urban_rural = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
#     urban_rural['FIPS'] = urban_rural['FIPS'].str.zfill(5)
#     data_df = pd.merge(
#         data_df,
#         urban_rural[['FIPS', '2023 Code']].rename(columns={'2023 Code': 'urban_rural_class'}),
#         on='FIPS',
#         how='left'
#     )
    
#     # Convert classes to 0-5 and drop missing
#     data_df['urban_rural_class'] = data_df['urban_rural_class'].astype(int) - 1
#     data_df = data_df.dropna(subset=['urban_rural_class'])
    
#     return data_df

def construct_data_df():
    """Builds a dataset with aggregated temporal features."""
    data_df = pd.DataFrame()

    # Process time-varying variables (e.g., unemployment, housing burden)
    variables = [v for v in DATA if v not in ['Mortality']]
    for i, variable in enumerate(variables):
        variable_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
        var_df = pd.read_csv(
            variable_path,
            dtype={'FIPS': str},
            header=0,
            names=['FIPS'] + [f'{year} {variable}' for year in range(2010, 2023)]
        )
        var_df['FIPS'] = var_df['FIPS'].str.zfill(5)

        # Melt to long format for temporal aggregation
        var_df_melted = var_df.melt(
            id_vars='FIPS',
            var_name='Year',
            value_name=variable
        )
        var_df_melted['Year'] = var_df_melted['Year'].str.extract('(\d+)').astype(int)
        
        def safe_slope(x):
            """Compute slope only if there are ≥2 data points and variance > 0."""
            if len(x) < 2 or np.var(x) == 0:
                return np.nan
            return np.polyfit(np.arange(len(x)), x, 1)[0]

        # Aggregate statistics
        aggregations = var_df_melted.groupby('FIPS').agg({
            variable: [
                ('mean', 'mean'),
                ('std', 'std'),
                ('slope', safe_slope),
                ('last', 'last')
            ]
        }).reset_index()
        
        # Flatten column names
        aggregations.columns = [
            'FIPS',
            f'{variable}_mean',
            f'{variable}_std',
            f'{variable}_slope',
            f'{variable}_last'
        ]
        
        # Initialize or merge
        if i == 0:  # First iteration: initialize data_df with FIPS and first variable
            data_df = aggregations.copy()
        else:       # Subsequent iterations: merge on FIPS
            data_df = pd.merge(data_df, aggregations, on='FIPS', how='left')
        
        # Drop rows with invalid slopes AFTER merging all variables (move this outside the loop)
        # data_df = data_df.dropna(subset=[f'{variable}_slope'])  # Remove this line here

    # Move the dropna step HERE (after all variables are merged)
    # Only drop rows where ALL slope columns are NaN (adjust as needed)
    #slope_columns = [f'{v}_slope' for v in variables]
    #data_df = data_df.dropna(subset=slope_columns, how='all')

    # Load and merge urban-rural classification
    urban_rural = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    urban_rural['FIPS'] = urban_rural['FIPS'].str.zfill(5)
    data_df = pd.merge(
        data_df,
        urban_rural[['FIPS', '2023 Code']].rename(columns={'2023 Code': 'urban_rural_class'}),
        on='FIPS',
        how='left'
    )
    
    # Convert classes to 0-5 and drop missing
    data_df['urban_rural_class'] = data_df['urban_rural_class'].astype(int) - 1
    #data_df = data_df.dropna(subset=['urban_rural_class'])
    
    # 3. Remove rows with ANY missing values in features
    feature_cols = [col for col in data_df.columns 
                   if any(x in col for x in ['mean', 'std', 'slope', 'last'])]
    data_df = data_df.dropna(subset=feature_cols)
    
    return data_df

def prepare_xgboost_data(data_df):
    # Split data into urban and rural subsets
    urban_mask = data_df['urban_rural_class'].isin([0, 1, 2, 3])
    rural_mask = data_df['urban_rural_class'].isin([4, 5])  # Original 5-6 becomes 4-5 after -1
    
    datasets = {
        'urban': {'data': data_df[urban_mask], 'classes': 4},
        'rural': {'data': data_df[rural_mask], 'classes': 2}
    }
    
    # Define features (same for both)
    features = [
        col for col in data_df.columns 
        if any(x in col for x in ['mean', 'std', 'slope', 'last'])#, 'Social Capital'])
    ]
    
    # Prepare splits for each category
    for category in datasets:
        subset = datasets[category]['data']
        X = subset[features]
        y = subset['urban_rural_class']
        
        # Convert rural classes to 0-1
        if category == 'rural':
            y = y - 4  # Map 4-5 → 0-1
        
        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Calculate class weights BEFORE SMOTE
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced', 
            classes=classes, 
            y=y_train  # Original y_train, not resampled
        )
        class_weights = {cls: weight for cls, weight in zip(classes, class_weights)}
        
        # Apply SMOTE only to training data
        smote = SMOTE(
            sampling_strategy='auto',  # Balances all minority classes
            k_neighbors=15,#min(3, len(y_train)-1),  # Safety for small classes
            random_state=42
        )
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        datasets[category]['X_train'] = X_train_res
        datasets[category]['X_test'] = X_test
        datasets[category]['y_train'] = y_train_res
        datasets[category]['y_test'] = y_test
        datasets[category]['class_weights'] = class_weights
        
        # Print class distribution
        print(f"\n{category.capitalize()} Class Distribution:")
        print(f"Original: {pd.Series(y).value_counts().sort_index()}")
        print(f"After SMOTE: {pd.Series(y_train_res).value_counts().sort_index()}")
    
    return datasets

def train_xgboost(X_train, y_train, X_test, y_test, num_classes, class_weights):
    
    if num_classes == 2:
        scale_pos_weight = class_weights[1] / class_weights[0]
        model = XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            early_stopping_rounds=10,
            random_state=42
        )
    else:
        model = XGBClassifier(
            class_weight=class_weights,
            objective="multi:softmax",
            num_class=num_classes,
            eval_metric="mlogloss",
            early_stopping_rounds=10,
            random_state=42
        )
        
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  # Critical fix
        verbose=False
    )
    return model

def main():
    data_df = construct_data_df()
    datasets = prepare_xgboost_data(data_df)
    
    # Urban classification (0-3 → Large Central Metro to Small Metro)
    urban_data = datasets['urban']
    urban_model = train_xgboost(
        urban_data['X_train'], urban_data['y_train'],
        urban_data['X_test'], urban_data['y_test'],
        num_classes=4,
        class_weights=datasets['urban']['class_weights']
    )
    urban_pred = urban_model.predict(urban_data['X_test'])
    print("\nUrban Classification Report:")
    print(classification_report(urban_data['y_test'], urban_pred, target_names=[
        "Large Central Metro", "Large Fringe Metro", 
        "Medium Metro", "Small Metro"
    ]))
    
    # Rural classification (0-1 → Micropolitan, Non-Core)
    rural_data = datasets['rural']
    rural_model = train_xgboost(
        rural_data['X_train'], rural_data['y_train'],
        rural_data['X_test'], rural_data['y_test'],
        num_classes=2,
        class_weights=datasets['rural']['class_weights']
    )
    rural_pred = rural_model.predict(rural_data['X_test'])
    print("\nRural Classification Report:")
    print(classification_report(rural_data['y_test'], rural_pred, target_names=[
        "Micropolitan", "Non-Core"
    ]))

if __name__ == "__main__":
    main()