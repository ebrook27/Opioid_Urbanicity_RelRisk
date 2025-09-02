### 3/27/25, EB: I tried the single-year XGB classifier on the Urban/Rural data, and it didn't work too well.
### I think the problem is not having a lot of training data. I'm going to try to include more years worth of the SVI data,
### and also include the Social Capital data for just 2018. I'm not sure how to do this yet, but this is my attempt.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
from xgboost import XGBClassifier
# import logging
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import RandomOverSampler


DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment', 'Social Capital', 'Social Connectedness']# 'Family Unity', 'Community Health',


# def construct_data_df():
#     """Constructs the data_df with full 6-class urban-rural codes."""
    
#     # Initialize with Mortality data (same as before)
#     mortality_path = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
#     data_df = pd.read_csv(
#         mortality_path,
#         header=0,
#         names=['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)],
#         dtype={'FIPS': str}
#     )
#     data_df['FIPS'] = data_df['FIPS'].str.zfill(5)

#     # Load other variables (unchanged)
#     for variable in [v for v in DATA if v != 'Mortality']:
#         variable_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
        
#         if variable == 'Social Capital':
#             sci_df = pd.read_csv(
#                 variable_path,
#                 usecols=['FIPS', '2018 Social Capital'],
#                 dtype={'FIPS': str}
#             )
#             sci_df['FIPS'] = sci_df['FIPS'].str.zfill(5)
#             data_df = pd.merge(data_df, sci_df, on='FIPS', how='left')
#         else:
#             var_df = pd.read_csv(
#                 variable_path,
#                 header=0,
#                 names=['FIPS'] + [f'{year} {variable}' for year in range(2010, 2023)],
#                 dtype={'FIPS': str}
#             )
#             var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
#             data_df = pd.merge(data_df, var_df, on='FIPS', how='left')

#     # Load NCHS data with 6-class codes
#     urban_rural = pd.read_csv(
#         'Data/SVI/NCHS_urban_v_rural.csv',
#         dtype={'FIPS': str},
#         usecols=['FIPS', '2023 Code']
#     )
#     urban_rural['FIPS'] = urban_rural['FIPS'].str.zfill(5)

#     # Merge and rename target column
#     data_df = pd.merge(
#         data_df,
#         urban_rural,
#         on='FIPS',
#         how='left'
#     ).rename(columns={'2023 Code': 'urban_rural_class'})


#     # Convert classes to 0-5 (if originally 1-6)
#     data_df['urban_rural_class'] = data_df['urban_rural_class'].astype(int) - 1  # Optional: adjust to 0-based

#     # print("Missing class labels:", data_df['urban_rural_class'].isna().sum())
#     # # Verify labels are 0-5
#     # print("Unique classes:", data_df['urban_rural_class'].unique())
    
#     return data_df

def construct_data_df():
    """Builds a dataset with aggregated temporal features."""
    # Load mortality and social capital (unchanged)
    mortality_path = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
    data_df = pd.read_csv(mortality_path, dtype={'FIPS': str})
    data_df['FIPS'] = data_df['FIPS'].str.zfill(5)

    # Load social capital (static 2018 data)
    sci_df = pd.read_csv(
        'Data/SVI/Final Files/Social Capital_final_rates.csv',
        usecols=['FIPS', '2018 Social Capital'],
        dtype={'FIPS': str}
    )
    sci_df['FIPS'] = sci_df['FIPS'].str.zfill(5)
    data_df = pd.merge(data_df, sci_df, on='FIPS', how='left')

    # Process time-varying variables (e.g., unemployment, housing burden)
    for variable in [v for v in DATA if v not in ['Mortality', 'Social Capital']]:
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
        
        # Compute aggregations
        # aggregations = var_df_melted.groupby('FIPS').agg({
        #     variable: [
        #         ('mean', np.mean),
        #         ('std', np.std),
        #         ('slope', lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]),
        #         ('last', lambda x: x.iloc[-1])
        #     ]
        # }).reset_index()
        
        ### 3/27/25, EB: Got a warning regarding the std function and np.polyfit, so I made the following changes:
        def safe_slope(x):
            """Compute slope only if there are ≥2 data points and variance > 0."""
            if len(x) < 2 or np.var(x) == 0:
                return np.nan  # Return NaN for invalid slopes
            return np.polyfit(np.arange(len(x)), x, 1)[0]

        # Update the aggregation to use safe_slope
        aggregations = var_df_melted.groupby('FIPS').agg({
            variable: [
                ('mean', 'mean'),
                ('std', 'std'),
                ('slope', safe_slope),  # Use the safeguarded function
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
        
        # Merge into main dataframe
        data_df = pd.merge(data_df, aggregations, on='FIPS', how='left')
        
        
        # # After merging aggregations into data_df:
        data_df = data_df.dropna(subset=[f'{variable}_slope'])  # Drop rows with invalid slopes
        # # OR
        #data_df[f'{variable}_slope'] = data_df[f'{variable}_slope'].fillna(0)  # Fill with 0

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
    data_df = data_df.dropna(subset=['urban_rural_class'])
    
    return data_df

# def prepare_xgboost_data(data_df, target_year=2020):
#     """Prepares features/target for a specific year."""
    
#     # Select features (SVI variables) for target_year
#     features = [
#         col for col in data_df.columns 
#         if str(target_year) in col and 'Mortality' not in col
#     ]
    
#     # Filter data for the target year and valid classes
#     df = data_df[['FIPS', 'urban_rural_class'] + features].dropna()
    
#     X = df[features]
#     y = df['urban_rural_class'].astype(int)  # Ensure classes are integers
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, 
#         test_size=0.2, 
#         stratify=y,  # Preserve class balance
#         random_state=42
#     )
    
#     # Upsample training data
#     oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
#     X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
#     print("Resampled class distribution:", pd.Series(y_train_resampled).value_counts())

#     # Calculate weights for the original training data
#     classes = np.unique(y_train)
#     weights = compute_class_weight(
#         class_weight='balanced', 
#         classes=classes, 
#         y=y_train_resampled
#     )
#     class_weights = dict(zip(classes, weights))
    
#     return X_train_resampled, X_test, y_train_resampled, y_test, class_weights

def prepare_xgboost_data(data_df):
    # Define features and target
    features = [
        col for col in data_df.columns 
        if any(x in col for x in ['mean', 'std', 'slope', 'last', 'Social Capital'])
    ]
    X = data_df[features]
    y = data_df['urban_rural_class']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


# def train_xgboost(X_train, y_train, X_test, y_test, class_weights):
#     """Trains XGBoost on 6-class data."""
    
#     OPTIMIZED_XGBOOST = XGBClassifier(
#         objective="multi:softmax",  # For multi-class classification
#         num_class=6,                # Number of classes (0-5)
#         eval_metric="mlogloss",     # Metric for evaluation
#         colsample_bytree=0.8,
#         gamma=0.1,
#         learning_rate=0.1,
#         max_depth=5,
#         n_estimators=300,
#         subsample=0.9,
#         random_state=42
#     )
    
    
#     # Create a weight array for the training data
#     sample_weights = np.array([class_weights[y] for y in y_train])
    
#     OPTIMIZED_XGBOOST.fit(
#         X_train, y_train,
#         sample_weight=sample_weights,  # Apply class weights
#         eval_set=[(X_test, y_test)],
#         verbose=False
#     )
    
#     return OPTIMIZED_XGBOOST

def train_xgboost(X_train, y_train, X_test, y_test):
    # Train XGBoost
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=6,
        eval_metric="mlogloss",
        early_stopping_rounds=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model

def main():
    # Prepare data
    data_df = construct_data_df()

    # Train model for 2020 (adjust year as needed)
    X_train, X_test, y_train, y_test = prepare_xgboost_data(data_df)
    model = train_xgboost(X_train, y_train, X_test, y_test)

    y_pred = model.predict(X_test)
    # Check predictions
    # print("Unique predicted classes:", np.unique(y_pred))  # Should output [0 1 2 3 4 5]
    print(classification_report(y_test, y_pred, target_names=[
       "Large Central Metro", "Large Fringe Metro", 
       "Medium Metro", "Small Metro",
       "Micropolitan", "Non-Core"
    ]))


if __name__ == "__main__":
    main()