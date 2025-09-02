import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import logging

log_file = 'Log Files/optimal_xgboost_parameters.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

xgb_model = xgb.XGBRegressor(
    objective='reg:absoluteerror', 
    random_state=42,
    tree_method='gpu_hist'  # Enables GPU acceleration
)

xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror', random_state=42)
KF = KFold(n_splits=5, shuffle=True, random_state=42)

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 
        # 'Disability', 
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
        'Single-Parent Household', 'Unemployed']

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

def features_targets(data_df, year):
    targets = data_df[f'{year} Mortality Rates']
    columns_to_keep = [f'{year-1} {feature} Rates' for feature in DATA if feature != 'Mortality']
    features = data_df[columns_to_keep]
    return features, targets

def optimize_xgb(xgb_model, features, targets):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 4, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 1],
        'subsample': [0.7, 0.8, 0.9],  # Subsample ratio of the training instances
        'colsample_bytree': [0.7, 0.8, 0.9],  # Subsample ratio of columns when constructing each tree
        'gamma': [0, 0.1, 0.2]  # Minimum loss reduction required to make a further partition on a leaf node
    }

    # Initialize the GridSearchCV with MAE as the scoring metric
    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='neg_mean_absolute_error', 
        verbose=2, 
        n_jobs=-1
    )
    grid_search.fit(features, targets)

    # Print the best parameters and best score
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best score found: {grid_search.best_score_}")

def main():
    data_df = construct_data_df()
    year = 2017
    features, targets = features_targets(data_df, year)
    optimize_xgb(xgb_model, features, targets)

if __name__ == "__main__":
    main()