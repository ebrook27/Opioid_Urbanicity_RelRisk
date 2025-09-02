### 3/27/25, EB: Here I'm trying to build a classifier for the augmented SVI dataset (plus Soc Cap and Soc Conn)
### to predict the urban vs rural character of counties in the US. I use the CDC's 6-category classification scheme.
### I need to figure out the temporal nature of the data, but I'll start with just 2018 to begin with.

### After testing initial performance, want to try weight classes to improve prediction.
### The weights seemed to help a little? Not a ton. Going to try oversampling the minority classes next.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler


DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment', 'Social Capital', 'Social Connectedness']# 'Family Unity', 'Community Health',


def construct_data_df():
    """Constructs the data_df with full 6-class urban-rural codes."""
    
    # Initialize with Mortality data (same as before)
    mortality_path = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
    data_df = pd.read_csv(
        mortality_path,
        header=0,
        names=['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)],
        dtype={'FIPS': str}
    )
    data_df['FIPS'] = data_df['FIPS'].str.zfill(5)

    # Load other variables (unchanged)
    for variable in [v for v in DATA if v != 'Mortality']:
        variable_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
        
        if variable == 'Social Capital':
            sci_df = pd.read_csv(
                variable_path,
                usecols=['FIPS', '2018 Social Capital'],
                dtype={'FIPS': str}
            )
            sci_df['FIPS'] = sci_df['FIPS'].str.zfill(5)
            data_df = pd.merge(data_df, sci_df, on='FIPS', how='left')
        else:
            var_df = pd.read_csv(
                variable_path,
                header=0,
                names=['FIPS'] + [f'{year} {variable}' for year in range(2010, 2023)],
                dtype={'FIPS': str}
            )
            var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
            data_df = pd.merge(data_df, var_df, on='FIPS', how='left')

    # Load NCHS data with 6-class codes
    urban_rural = pd.read_csv(
        'Data/SVI/NCHS_urban_v_rural.csv',
        dtype={'FIPS': str},
        usecols=['FIPS', '2023 Code']
    )
    urban_rural['FIPS'] = urban_rural['FIPS'].str.zfill(5)

    # Merge and rename target column
    data_df = pd.merge(
        data_df,
        urban_rural,
        on='FIPS',
        how='left'
    ).rename(columns={'2023 Code': 'urban_rural_class'})


    # Convert classes to 0-5 (if originally 1-6)
    data_df['urban_rural_class'] = data_df['urban_rural_class'].astype(int) - 1  # Optional: adjust to 0-based

    # print("Missing class labels:", data_df['urban_rural_class'].isna().sum())
    # # Verify labels are 0-5
    # print("Unique classes:", data_df['urban_rural_class'].unique())
    
    return data_df

def prepare_xgboost_data(data_df, target_year=2020):
    """Prepares features/target for a specific year."""
    
    # Select features (SVI variables) for target_year
    features = [
        col for col in data_df.columns 
        if str(target_year) in col and 'Mortality' not in col
    ]
    
    # Filter data for the target year and valid classes
    df = data_df[['FIPS', 'urban_rural_class'] + features].dropna()
    
    X = df[features]
    y = df['urban_rural_class'].astype(int)  # Ensure classes are integers
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y,  # Preserve class balance
        random_state=42
    )
    
    # Upsample training data
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
    print("Resampled class distribution:", pd.Series(y_train_resampled).value_counts())

    # # Calculate weights for the original training data
    # classes = np.unique(y_train)
    # weights = compute_class_weight(
    #     class_weight='balanced', 
    #     classes=classes, 
    #     y=y_train_resampled
    # )
    # class_weights = dict(zip(classes, weights))
    
    return X_train_resampled, X_test, y_train_resampled, y_test#, class_weights

def train_xgboost(X_train, y_train, X_test, y_test):#, class_weights):
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
    
    
    # # Create a weight array for the training data
    # sample_weights = np.array([class_weights[y] for y in y_train])
    
    OPTIMIZED_XGBOOST.fit(
        X_train, y_train,
        #sample_weight=sample_weights,  # Apply class weights
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return OPTIMIZED_XGBOOST



def main():
    # Prepare data
    data_df = construct_data_df()

    # Train model for 2020 (adjust year as needed)
    X_train, X_test, y_train, y_test = prepare_xgboost_data(data_df, target_year=2018)
    model = train_xgboost(X_train, y_train, X_test, y_test)#, class_weights)

    # Evaluate
    from sklearn.metrics import classification_report
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