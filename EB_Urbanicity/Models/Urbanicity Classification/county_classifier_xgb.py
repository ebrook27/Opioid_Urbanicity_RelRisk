### 3/28/25, EB: So we're dropping Social Capital and Connectedness for now. Going to just focus on the "SVI" variables to predict county urban vs rural classificaiton. The good thing about this 
### is that they contain multiple years worth of data. The complication with this comes from trying to use them with XGB classifier. The XGB classifier requires a single target variable, 
### but the SVI variables are all time series data. I'm going to try summary stats for the SVI variables first, and ask ChatGPT for any other ideas. Might have to abandon this approach
### for classifying the counties.


### 3/30/25, EB: Tried to get this working with simple summary stats, added several more summary stats, and it's still predicting poorly. XGB just can't handle the temporal data. I could try to use
### maybe an ensemble XGB? I think ultimately XGB is just not the move here, so I'm moving on (for now) to a MLP model. I think it will be able to capture the temporal data better, and I might use an RNN to beef it up.
### Going to have to experiment.

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
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']


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

# def construct_data_df():
#     """Builds a dataset with aggregated temporal features."""
#     ### 3/28/25, EB: Dropping Mortality data from prediction. Don't want to double-dip.
    
#     # # Load mortality and social capital (unchanged)
#     # mortality_path = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
#     # data_df = pd.read_csv(mortality_path, dtype={'FIPS': str})
#     # data_df['FIPS'] = data_df['FIPS'].str.zfill(5)
#     data_df = pd.DataFrame()

#     # Process time-varying variables (e.g., unemployment, housing burden)
#     for variable in [v for v in DATA if v not in ['Mortality']]:
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
        how='inner' #CHANGED FROM LEFT TO INNER TO TEST SMOTE
    )
    
    # Convert classes to 0-5 and drop missing
    data_df['urban_rural_class'] = data_df['urban_rural_class'].astype(int) - 1
    #data_df = data_df.dropna(subset=['urban_rural_class'])
    
    # 3. Remove rows with ANY missing values in features
    feature_cols = [col for col in data_df.columns 
                   if any(x in col for x in ['mean', 'std', 'slope', 'last'])]
    data_df = data_df.dropna(subset=feature_cols)
    
    return data_df

# def construct_data_df():
#     """Builds a dataset with aggregated temporal features with enhanced statistics."""
#     data_df = pd.DataFrame()

#     # Process time-varying variables (e.g., unemployment, housing burden)
#     variables = [v for v in DATA if v not in ['Mortality']]
    
#     for i, variable in enumerate(variables):
#         variable_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
#         var_df = pd.read_csv(
#             variable_path,
#             dtype={'FIPS': str},
#             header=0,
#             names=['FIPS'] + [f'{year} {variable}' for year in range(2010, 2023)]
#         )
#         var_df['FIPS'] = var_df['FIPS'].str.zfill(5)

#         # Melt to long format and ensure temporal ordering
#         var_df_melted = var_df.melt(
#             id_vars='FIPS',
#             var_name='Year',
#             value_name=variable
#         )
#         var_df_melted['Year'] = var_df_melted['Year'].str.extract('(\d+)').astype(int)
#         var_df_melted = var_df_melted.sort_values(['FIPS', 'Year'])
        
#         # Define custom aggregation functions
#         def safe_slope(x):
#             """Linear slope with validation."""
#             if len(x) < 2 or np.var(x) == 0:
#                 return np.nan
#             return np.polyfit(np.arange(len(x)), x, 1)[0]
        
#         def safe_autocorr(x):
#             """Lag-1 autocorrelation with validation."""
#             if len(x) < 2 or np.var(x) == 0:
#                 return np.nan
#             return pd.Series(x).autocorr(lag=1)
        
#         def safe_acceleration(x):
#             """Quadratic acceleration (2nd derivative) with validation."""
#             if len(x) < 3 or np.var(x) == 0:
#                 return np.nan
#             return 2 * np.polyfit(np.arange(len(x)), x, 2)[0]  # 2 * quadratic coefficient

#         # Aggregate statistics with new features
#         aggregations = var_df_melted.groupby('FIPS').agg({
#             variable: [
#                 ('mean', 'mean'),
#                 ('std', 'std'),
#                 ('slope', safe_slope),
#                 ('last', 'last'),
#                 ('min', 'min'),
#                 ('max', 'max'),
#                 ('q25', lambda x: x.quantile(0.25)),
#                 ('q75', lambda x: x.quantile(0.75)),
#                 ('autocorr', safe_autocorr),
#                 ('acceleration', safe_acceleration)
#             ]
#         }).reset_index()
        
#         # Flatten column names
#         aggregations.columns = [
#             'FIPS',
#             f'{variable}_mean',
#             f'{variable}_std',
#             f'{variable}_slope',
#             f'{variable}_last',
#             f'{variable}_min',
#             f'{variable}_max',
#             f'{variable}_q25',
#             f'{variable}_q75',
#             f'{variable}_autocorr',
#             f'{variable}_acceleration'
#         ]
        
#         # Add derived range feature
#         aggregations[f'{variable}_range'] = aggregations[f'{variable}_max'] - aggregations[f'{variable}_min']
        
#         # Initialize or merge
#         if i == 0:
#             data_df = aggregations.copy()
#         else:
#             data_df = pd.merge(data_df, aggregations, on='FIPS', how='left')

#     # Handle missing values (now checking all new features)
#     temporal_features = [f'{v}_{stat}' for v in variables 
#                         for stat in ['slope', 'autocorr', 'acceleration']]
#     data_df = data_df.dropna(subset=temporal_features, how='all')

#     # Merge urban-rural classification
#     urban_rural = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
#     urban_rural['FIPS'] = urban_rural['FIPS'].str.zfill(5)
#     data_df = pd.merge(
#         data_df,
#         urban_rural[['FIPS', '2023 Code']].rename(columns={'2023 Code': 'urban_rural_class'}),
#         on='FIPS',
#         how='inner'  # Use inner join to automatically drop counties without classification
#     )
    
#     # Final class adjustment
#     data_df['urban_rural_class'] = (data_df['urban_rural_class'] >= 5).astype(int)#data_df['urban_rural_class'].astype(int) - 1
    
#     return data_df

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
        if any(x in col for x in ['mean', 'std', 'slope', 'last'])
    ]
    #print("Features selected:", features)  # Debugging line
    X = data_df[features]
    y = data_df['urban_rural_class']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


# def train_xgboost(X_train, y_train, X_test, y_test, class_weights):
    """Trains XGBoost on 6-class data."""
    
    OPTIMIZED_XGBOOST = XGBClassifier(
        objective="multi:softmax",  # For multi-class classification
        num_class=6,                # Number of classes (0-5)
        eval_metric="mlogloss",     # Metric for evaluation
        colsample_bytree=0.8,
        gamma=0.1,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=300,
        subsample=0.9,
        random_state=42
    )
    
    
    # Create a weight array for the training data
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    OPTIMIZED_XGBOOST.fit(
        X_train, y_train,
        sample_weight=sample_weights,  # Apply class weights
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return OPTIMIZED_XGBOOST

# def train_xgboost(X_train, y_train, X_test, y_test):
#     # Train XGBoost
#     model = XGBClassifier(
#         objective="multi:softmax",
#         num_class=6,
#         eval_metric="mlogloss",
#         #early_stopping_rounds=10,
#         random_state=42
#     )
#     model.fit(X_train, y_train)

#     return model

def train_xgboost(X_train, y_train, X_test, y_test):
    # This function trains an XGBoost model on the provided training data.
    # It uses SMOTE for oversampling the minority class and applies class weights.
    
    # Apply SMOTE only to training data
    ################################################################################################################################
    # ## 4/1/25, EB:
    # ## Tried SMOTE with random_state=42, k_neighbors=5, but it didn't work GREAT. Trying a more targeted oversamplign based on f-scores
    # #smote = SMOTE(random_state=42, k_neighbors=5)
    # # Calculate class weights based on F1-scores
    # f1_scores = [0.72, 0.57, 0.38, 0.27, 0.42, 0.77]  # From classification report
    # # Calculate inverse F1-scores (lower F1 → higher weight)
    # inverse_f1 = [1 / (score + 1e-5) for score in f1_scores]  # +1e-5 to avoid division by zero

    # # Normalize weights to determine oversampling ratios
    # total_weight = sum(inverse_f1)
    # class_ratios = {i: weight / total_weight for i, weight in enumerate(inverse_f1)}

    # # Get original class counts
    # original_counts = y_train.value_counts().sort_index()

    # # Calculate target counts (ensure ≥ original count)
    # sampling_strategy = {}
    # for cls in original_counts.index:
    #     target = int(len(y_train) * class_ratios[cls])
    #     sampling_strategy[cls] = max(target, original_counts[cls])  # Critical line
        
    # # Now apply SMOTE
    # smote = SMOTE(
    #     sampling_strategy=sampling_strategy,
    #     k_neighbors=5,
    #     random_state=42
    # )
    # X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # # Check class distribution after SMOTE
    # print("\nClass distribution after SMOTE:")
    # print(pd.Series(y_train_res).value_counts().sort_index())
    ###########################################################################################################################
    # ### 4/1/25, EB: Things are a bit better now, but I want to compare to ADASYN for just the 3 middle classes. I think it might be better.
    # target_classes=[3]
    # # Get original class distribution
    # original_counts = y_train.value_counts().sort_index()
    # print("Original class distribution:\n", original_counts)

    # # Define sampling strategy for ADASYN
    # sampling_strategy = {cls: count * 5  # Triple samples for target classes
    #                     for cls, count in original_counts.items()
    #                     if cls in target_classes}

    # # Initialize ADASYN
    # adasyn = ADASYN(
    #     sampling_strategy=sampling_strategy,
    #     n_neighbors=5,  # Reduce if small class sizes
    #     random_state=42
    # )

    # # Apply ADASYN
    # X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
    ###########################################################################################################################
    # ### 4/1/25, EB: ADASYN worked ok, but going to try BorderlineSMOTE with just the categories 3 and 4.
    # # Target only class 4 (Micropolitan)
    # bsmote = BorderlineSMOTE(
    #     sampling_strategy={3:600, 4: 600},  # Generate 300 samples for class 4
    #     k_neighbors=5,
    #     m_neighbors=10,
    #     kind='borderline-1',
    #     random_state=42
    # )
    # # Apply BorderlineSMOTE
    # X_train_res, y_train_res = bsmote.fit_resample(X_train, y_train)
    ###########################################################################################################################
    ### 4/1/25, EB:Ok, nothing is really working, so I'm going to try both oversampling the poor classes and undersampling the better classes. 
    ### I think this might work?
    f1_scores = [0.50, 0.56, 0.41, 0.29, 0.47, 0.77]
    
    underperforming = [i for i, score in enumerate(f1_scores) if score < 0.65]  # Classes 2,4
    well_predicted = [i for i, score in enumerate(f1_scores) if score >= 0.7]   # Classes 0,1,3,5
    
    # Calculate original counts
    original_counts = y_train.value_counts().sort_index()

    # Oversample underperformers: Triple their size
    oversample_strategy = {cls: 3 * original_counts[cls] for cls in underperforming}

    # Undersample well-predicted: Halve their size
    undersample_strategy = {cls: original_counts[cls] // 2 for cls in well_predicted}

    # Combine strategies
    pipeline = Pipeline([
        ('oversample', SMOTE(sampling_strategy=oversample_strategy, k_neighbors=5)),
        ('undersample', RandomUnderSampler(sampling_strategy=undersample_strategy))
    ])
    
    X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
    
    # Train XGBoost on resampled data
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=6,
        eval_metric="mlogloss",
        random_state=42,
        scale_pos_weight='sum'  # Additional class balancing
    )
    model.fit(X_train_res, y_train_res)
    
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
    # print(classification_report(y_test, y_pred, target_names=[
    #    "Urban", "Rural"
    # ]))


if __name__ == "__main__":
    main()