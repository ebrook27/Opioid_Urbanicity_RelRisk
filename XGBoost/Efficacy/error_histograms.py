import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} MR' for year in range(2010, 2023)]

def load_mortality_rates(data_path, data_names):
    mort_df = pd.read_csv(data_path, header=0, names=data_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    mort_df[data_names[1:]] = mort_df[data_names[1:]].astype(float).clip(lower=0)
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

def construct_histogram(mort_df, preds_df, year):
    errors = abs(preds_df[f'{year} Preds'] - mort_df[f'{year} MR'])
    max_error = round(errors.max(), 2)

    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('XGBoost Absolute Error', fontsize=12, weight='bold')
    plt.ylabel('Frequency', fontsize=12, weight='bold')

    if year >= 2021:
        tick_positions = np.arange(0, max_error+20, 20)
    else:    
        tick_positions = np.arange(0, max_error+10, 10)

    tick_labels = [str(int(x)) for x in tick_positions]
    plt.xticks(tick_positions, tick_labels) 

    title = f'{year} XGBoost Error Histogram'
    plt.title(title, size=16, weight='bold')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f'XGBoost/Efficacy/Error Histograms/{year}_xgb_err_histo', bbox_inches=None, pad_inches=0, dpi=300)
    #plt.show()
    plt.close()

def main():
    mort_df = load_mortality_rates(MORTALITY_PATH, MORTALITY_NAMES)
    preds_df = load_yearly_predictions()

    for year in range(2011, 2023):
        construct_histogram(mort_df, preds_df, year)
        print(f'Histo printed for {year}.')

if __name__ == "__main__":
    main()

