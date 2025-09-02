import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import xgboost as xgb
import logging

# Constants
NUM_YEARS = len(range(2010,2023))
OPTIMIZED_XGBOOST = xgb.XGBRegressor(
    colsample_bytree=0.8,
    gamma=0.1,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=300,
    subsample=0.9,
    random_state=42,
    objective='reg:absoluteerror'  # Set the loss function to MAE
    )
KF = KFold(n_splits=5, shuffle=True, random_state=42)
DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 
        # 'Disability', 
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
        'Single-Parent Household', 'Unemployment']

# Set up logging
log_file = 'Log Files/xgboost_model.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def construct_data_df():
    data_df = pd.DataFrame()
    for variable in DATA:
        if variable == 'Mortality':
            variable_path = f'Data/Mortality/Final Files/{variable}_final_rates.csv'
        else:
            variable_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'

        variable_names = ['FIPS'] + [f'{year} {variable} Rates' for year in range(2010, 2023)]
        variable_df = pd.read_csv(variable_path, header=0, names=variable_names)
        variable_df['FIPS'] = variable_df['FIPS'].astype(str).str.zfill(5)
        variable_df[variable_names[1:]] = variable_df[variable_names[1:]].astype(float)

        if data_df.empty:
            data_df = variable_df
        else:
            data_df = pd.merge(data_df, variable_df, on='FIPS', how='outer')

    data_df = data_df.sort_values(by='FIPS').reset_index(drop=True)
    return data_df

def strip_fips(data_df):
    fips_codes = data_df['FIPS']
    return fips_codes

def features_targets(data_df, year):
    targets = data_df[f'{year} Mortality Rates']
    columns_to_keep = [f'{year-1} {feature} Rates' for feature in DATA if feature != 'Mortality']
    features = data_df[columns_to_keep]
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

def save_predictions(year, xgb_predictions, test_fips):
    saving_df = pd.DataFrame({
        'FIPS': test_fips,
        f'{year} XGBoost Predictions': xgb_predictions,
    })
    saving_df[f'{year} XGBoost Predictions'] = saving_df[f'{year} XGBoost Predictions'].round(2)
    saving_df = saving_df.sort_values(by='FIPS').reset_index(drop=True)
    saving_df.to_csv(f'XGBoost/XGBoost Predictions/{year}_xgboost_predictions.csv', index=False)

def update_total_importance(feature_importances, total_importance):
    total_importance += feature_importances
    return total_importance

def plot_feature_importance(feature_importance_df):
    # Calculate the average importance across all years
    feature_importance_df['Average'] = feature_importance_df.mean(axis=1)

    # Sort the DataFrame by the average importance
    feature_importance_df = feature_importance_df.sort_values(by='Average', ascending=True)

    # Get the variables (features) and years (columns)
    features = feature_importance_df.index
    years = feature_importance_df.columns

    # Define bar width and positions for each group
    bar_width = 0.6
    y_positions = np.arange(len(features))  # Spacing between feature groups

    # Define colors
    num_years = len(years) - 1  # Exclude 'Average'
    colors = list(plt.cm.tab20.colors[:num_years]) + ['black']  # Add black for 'Average'

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each year's bars
    for i, year in enumerate(years):
        ax.barh(y_positions - i * bar_width / num_years, feature_importance_df[year],
                height=bar_width / num_years, label=year, color=colors[i])

    # Adjust labels, title, and legend
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features, fontsize=20)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)  # Increase the font size of x-axis tick labels
    ax.set_title('XGBoost Feature Importance', fontsize=20, fontweight='bold')
    ax.legend(title='Year', fontsize=15, title_fontsize=15, loc='lower right')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('Feature Importance/xgboost_feature_importance.png', bbox_inches='tight')
    plt.close()

    # Log the average feature importance for each variable
    feature_importance_df = feature_importance_df.sort_values(by='Average', ascending=False)
    logging.info("Average Feature Importance for each variable:")
    for feature, avg_importance in feature_importance_df['Average'].items():
        logging.info(f"{feature}: {avg_importance:.4f}")

def main():
    yearly_importance_dict = {yr: [] for yr in range(2011, 2023)}
    num_features = len(DATA) - 1
    total_importance = np.zeros(num_features)
    data_df = construct_data_df()
    fips_codes = strip_fips(data_df)
    for year in range(2011, 2023): # We start predicting for 2011, not 2010
        features, targets = features_targets(data_df, year)
        feature_importances, xgb_predictions, test_fips = run_xgboost(features, targets, fips_codes)
        save_predictions(year, xgb_predictions, test_fips)
        yearly_importance_dict[year] = feature_importances.tolist()
        total_importance = update_total_importance(feature_importances, total_importance)
    total_importance = total_importance / NUM_YEARS

    # Final overall importance plot
    feature_names = [feature for feature in DATA if feature != 'Mortality']
    feature_importance_df = pd.DataFrame(yearly_importance_dict, index=feature_names)
    feature_importance_df['Average'] = total_importance
    feature_importance_df = feature_importance_df.sort_values('Average', ascending=True)
    plot_feature_importance(feature_importance_df)

if __name__ == "__main__":
    main()