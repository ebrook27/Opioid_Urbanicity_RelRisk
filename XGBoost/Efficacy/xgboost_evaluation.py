import pandas as pd
import numpy as np
import logging

# Constants
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} MR' for year in range(2010, 2023)]
YEAR = 2022 # for printing top errors

# Set up logging
log_file = 'Log Files/xgboost_efficacy.log'
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def load_mortality(mort_path, mort_names):
    mort_df = pd.read_csv(mort_path, header=0, names=mort_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[mort_names[1:]] = mort_df[mort_names[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def load_yearly_predictions():
    preds_df = pd.DataFrame()
    for year in range(2011,2023):
        yearly_path = f'XGBoost/XGBoost Predictions/{year}_xgboost_predictions.csv'
        yearly_names = ['FIPS'] + [f'{year} Preds']
        yearly_df = pd.read_csv(yearly_path, header=0, names=yearly_names)
        yearly_df['FIPS'] = yearly_df['FIPS'].astype(str).str.zfill(5)
        yearly_df[f'{year} Preds'] = yearly_df[f'{year} Preds'].astype(float)

        if preds_df.empty:
            preds_df = yearly_df
        else:
            preds_df = pd.merge(preds_df, yearly_df, on='FIPS', how='outer')

    preds_df = preds_df.sort_values(by='FIPS').reset_index(drop=True)
    return preds_df

def calculate_efficacy_metrics(mort_df, preds_df):
    acc_df = mort_df[['FIPS']].copy()
    metrics = {'Year': [], 'Avg Error': [], 'Max Error': [], 'Avg Accuracy': [], 
               'MSE': [], 'R2': [], 'MedAE': []}

    for year in range(2011, 2023):
        absolute_errors = abs(preds_df[f'{year} Preds'] - mort_df[f'{year} MR'])
        acc_df[f'{year} Absolute Errors'] = absolute_errors
        avg_err = np.mean(absolute_errors)
        max_err = absolute_errors.max()
        mse = np.mean(absolute_errors ** 2)
        r2 = 1 - (np.sum((mort_df[f'{year} MR'] - preds_df[f'{year} Preds']) ** 2) / np.sum((mort_df[f'{year} MR'] - np.mean(mort_df[f'{year} MR'])) ** 2))
        medae = np.median(absolute_errors)

        # Adjusting accuracy calculation
        if max_err == 0:  # Perfect match scenario
            acc_df[f'{year} Accuracy'] = 0.9999
        else:
            acc_df[f'{year} Accuracy'] = 1 - (absolute_errors / max_err)
            acc_df[f'{year} Accuracy'] = acc_df[f'{year} Accuracy'].apply(lambda x: 0.9999 if x == 1 else (0.0001 if x == 0 else x))
        
        avg_acc = np.mean(acc_df[f'{year} Accuracy'])
        
        metrics['Year'].append(year)
        metrics['Avg Error'].append(avg_err)
        metrics['Max Error'].append(max_err)
        metrics['Avg Accuracy'].append(avg_acc)
        metrics['MSE'].append(mse)
        metrics['R2'].append(r2)
        metrics['MedAE'].append(medae)
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df['Avg Error'] = metrics_df['Avg Error'].round(2)  # round to 2 decimal places
    metrics_df['Max Error'] = metrics_df['Max Error'].round(2)  # round to 2 decimal places
    metrics_df['Avg Accuracy'] = (metrics_df['Avg Accuracy'] * 100).round(2).astype(str) + '%'  # multiply by 100, round to 2 decimal places, and add % sign
    metrics_df['MSE'] = metrics_df['MSE'].round(2)  # round to 2 decimal places
    metrics_df['R2'] = metrics_df['R2'].round(2)  # round to 2 decimal places
    metrics_df['MedAE'] = metrics_df['MedAE'].round(2)  # round to 2 decimal places
    logging.info(metrics_df)

def print_top_errors(mort_df, preds_df, year=YEAR):
    err_df = mort_df[['FIPS']].copy()
    err_df[f'{year} Absolute Errors'] = abs(preds_df[f'{year} Preds'] - mort_df[f'{year} MR'])

    top_errors_df = err_df.sort_values(by=f'{year} Absolute Errors', ascending=False).head(10)
    top_errors_df_reset = top_errors_df.reset_index(drop=True)
    logging.info(f'\n{top_errors_df_reset}')

def main():
    mort_df = load_mortality(MORTALITY_PATH, MORTALITY_NAMES)
    preds_df = load_yearly_predictions()
    calculate_efficacy_metrics(mort_df, preds_df)
    print_top_errors(mort_df, preds_df)

if __name__ == "__main__":
    main()