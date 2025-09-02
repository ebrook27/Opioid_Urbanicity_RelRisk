## EB, 3/18/25
## Here I am investigating the feature importance calculation in the XGBoost model when 
## we include the social capital index and then try to include the social connectedness index.
## The difficulty is going to come from the fact that the social capital index is for the year 2018,
## rather than all years considered here. I'm going to start by just calculating the feature importance
## rankings for the year 2018, and then think about how that's changed over time.
## If I can get that working correctly, then I will try to include the social connectedness index.

## EB, 3/19/25
## I was able to get the social capital index included correctly, but only for the year 2018.
## Now I'm trying to include the social connectedness index, which I also only have a single year for.
## In the documentation for the social connectedness index, they say that the score shouldn't change
## much at all from year to year, so I'm going to include it for the year 2018, same as social capital.

## EB, 3/20/25
## The Social Connectedness index seems a little fishy, its importance is over twice the next highest feature.
## I also realized that the Social Capital data contained both positive and negative values, and since Andrew
## had transformed every feature into relative rates, I realized I needed to make the data match. I did this by doing a 
## min-max normalization on the Social Capital data.
## The Social Capital data is composed of four sub-indices, so we could investigate those individually as well. I've now removed
## Social Capital and Social Connectedness, and replaced them by Family Unity (it has the least amount of missing counties).
## Going to see how this performs and go from there. 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import xgboost as xgb
import logging

# Constants - Single year focus
TARGET_YEAR = 2018
OPTIMIZED_XGBOOST = xgb.XGBRegressor(
    colsample_bytree=0.8,
    gamma=0.1,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=300,
    subsample=0.9,
    random_state=42,
    objective='reg:absoluteerror'
)
KF = KFold(n_splits=5, shuffle=True, random_state=42)
DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment', 'Family Unity', 'Community Health', 'Social Capital', 'Social Connectedness']

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(f'Log Files/xgboost_{TARGET_YEAR}.log', mode='w'),
    logging.StreamHandler()
])

def construct_data_df():
    # Initialize with Mortality data
    mortality_path = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
    data_df = pd.read_csv(
        mortality_path,
        header=0,
        names=['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)],
        dtype={'FIPS': str}
    )
    data_df['FIPS'] = data_df['FIPS'].str.zfill(5)

    # Load other variables
    for variable in [v for v in DATA if v != 'Mortality']:
        variable_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
        
        # Handle Social Capital (2018 only)
        if variable == 'Social Capital':
            sci_df = pd.read_csv(
                variable_path,
                usecols=['FIPS', '2018 Social Capital'],  # Explicitly select 2018 column
                dtype={'FIPS': str}
            )
            sci_df['FIPS'] = sci_df['FIPS'].str.zfill(5)
            print(sci_df.head)
            data_df = pd.merge(data_df, sci_df, on='FIPS', how='left')
        else:
            # Load other variables with all years
            var_df = pd.read_csv(
                variable_path,
                header=0,
                names=['FIPS'] + [f'{year} {variable}' for year in range(2010, 2023)],
                dtype={'FIPS': str}
            )
            var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
            data_df = pd.merge(data_df, var_df, on='FIPS', how='left')

    return data_df

def features_targets(data_df):
    """Match your actual column names"""
    # Features: All 2018 SVI columns (without "Rates")
    features = data_df[[
        '2018 Aged 17 or Younger',
        '2018 Aged 65 or Older',
        '2018 Below Poverty',
        '2018 Crowding',
        '2018 Group Quarters',
        '2018 Limited English Ability',
        '2018 Minority Status',
        '2018 Mobile Homes',
        '2018 Multi-Unit Structures',
        '2018 No High School Diploma',
        '2018 No Vehicle',
        '2018 Single-Parent Household',
        '2018 Unemployment',
        '2018 Family Unity',  # Replaced '2018 Social Capital' with '2018 Family Unity'
        '2018 Social Capital',  # No "Rates" suffix
        # '2018 Community Health',
        '2018 Social Connectedness'
    ]]
    
    # Target: Mortality column retains "Rates" suffix
    targets = data_df['2019 Mortality Rates']
    
    return features, targets

def run_xgboost(features, targets, fips_codes):
    xgb_predictions = []
    all_test_fips = []  # Collect FIPS codes for all test sets
    feature_importances = np.zeros(features.shape[1])
    
    for train_index, test_index in KF.split(features):
        # Splitting the data for this fold
        train_features, test_features = features.iloc[train_index], features.iloc[test_index]
        train_targets, test_targets = targets.iloc[train_index], targets.iloc[test_index]
        test_fips = fips_codes.iloc[test_index]  # Get FIPS codes for the test set

        # Training the XGBoost model
        OPTIMIZED_XGBOOST.fit(train_features, train_targets)
        predictions = OPTIMIZED_XGBOOST.predict(test_features)
        
        # Update with the results from the current fold
        feature_importances += OPTIMIZED_XGBOOST.feature_importances_
        xgb_predictions.extend(predictions)
        all_test_fips.extend(test_fips)  # accumulate FIPS codes in the order they are used to test

    feature_importances = feature_importances / KF.get_n_splits()

    return feature_importances, xgb_predictions, all_test_fips


def main():
    # Load and prepare data
    data_df = construct_data_df()
    #Debug
    # print(data_df['2018 Social Capital'].head)
    # print(data_df['2018 Social Connectedness'].head)
    
    # Clean data: Remove rows with missing target or features
    data_df = data_df.dropna(subset=[f'2019 Mortality Rates'])
    # data_df['2018 Social Capital'] = data_df['2018 Social Capital'].fillna(0)
    # data_df['2018 Social Connectedness'] = data_df['2018 Social Connectedness'].fillna(0)
    # #print("Social Capital non-NaN count:", data_df['2018 Social Capital'].notna().sum())
    # data_df = data_df.dropna(axis=1, how='all')  # Remove empty columns
    #Debug
    #print(data_df.head)
    
    # Get features and targets
    features, targets = features_targets(data_df)
    fips_codes = data_df['FIPS']
    #print(features['2018 Social Capital'])
    
    # # Validate Social Capital exists
    # sci_col = '2018 Social Capital Rates'
    # assert sci_col in features.columns, f"Missing {sci_col} in features!"
    
    # Run model
    feature_importances, predictions, test_fips = run_xgboost(features, targets, fips_codes)
    
    # Save results
    pd.DataFrame({
        'FIPS': test_fips,
        'Prediction': predictions
    }).to_csv(f'XGBoost/2019_predictions.csv', index=False)
    
    # Plot feature importance
    importance_df = pd.DataFrame({
        'Feature': [c.replace('2019 ', '').replace(' Rates', '') for c in features.columns],
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='darkblue')
    plt.title('2019 Feature Importance (Using 2018 Features)')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('Feature Importance/2019_feature_importance_with_connectedness.png')
    plt.close()
    
    logging.info("2019 Feature Importance:\n" + importance_df.to_string(index=False))

if __name__ == "__main__":
    main()