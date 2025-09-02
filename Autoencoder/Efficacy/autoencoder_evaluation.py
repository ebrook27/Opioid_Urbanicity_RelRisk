from scipy.stats import lognorm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Constants
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} MR' for year in range(2010, 2023)]
PREDICTIONS_PATH = 'Autoencoder/Predictions/ae_mortality_predictions.csv'
PREDICTIONS_NAMES = [f'{year} Preds' for year in range(2011, 2023)]
YEAR = 2022 # for printing top errors

# Set up logging
log_file = 'Log Files/autoencoder_efficacy.log'
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

def load_predictions(preds_path=PREDICTIONS_PATH, preds_names=PREDICTIONS_NAMES):
    preds_df = pd.read_csv(preds_path, header=0, names=preds_names)
    preds_df[preds_names] = preds_df[preds_names].astype(float)
    preds_df = preds_df.reset_index(drop=True)
    return preds_df

def load_predictions_alternate(preds_path=PREDICTIONS_PATH, preds_names=PREDICTIONS_NAMES):
    preds_df = pd.read_csv(preds_path, header=0, names=preds_names)
    preds_df[preds_names] = preds_df[preds_names].astype(float)

    # Initialize dictionaries to store the predicted means and standard deviations
    predicted_shapes = {}
    predicted_locs = {}
    predicted_scales = {}
    start_year = 2011

    # Extract the last three rows (shape, location, scale)
    for i, col in enumerate(preds_names):
        year = start_year + i
        predicted_shapes[year] = preds_df[col].iloc[-3]
        predicted_locs[year] = preds_df[col].iloc[-2]
        predicted_scales[year] = preds_df[col].iloc[-1]

    # Drop the last three rows from the predicted rates
    preds_df = preds_df.iloc[:-3].reset_index(drop=True)
    return preds_df, predicted_shapes, predicted_locs, predicted_scales

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

def kl_divergence_lognorm(shape1, loc1, scale1, shape2, loc2, scale2):
    sigma1, mu1 = shape1, np.log(scale1)
    sigma2, mu2 = shape2, np.log(scale2)
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

def wasserstein_distance_lognorm(shape1, loc1, scale1, shape2, loc2, scale2):
    sigma1, mu1 = shape1, np.log(scale1)
    sigma2, mu2 = shape2, np.log(scale2)
    return np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)

def compare_distributions(mort_df, predicted_shapes, predicted_locs, predicted_scales):
    # Create an empty list to store the results
    results = {'Year': [], 'KL Divergence': [], 'Wasserstein Distance': []}

    # Loop through each year to compare actual and predicted distributions
    for year in range(2011, 2023):
        mort_rates = mort_df[f'{year} MR'].values
        non_zero_mort_rates = mort_rates[mort_rates > 0]
        lognorm_params = lognorm.fit(non_zero_mort_rates)
        shape, loc, scale = lognorm_params
    
        predicted_shape = predicted_shapes[year]
        predicted_loc = predicted_locs[year]
        predicted_scale = predicted_scales[year]

        # Compute the KL divergence and Wasserstein distance
        kl_div = kl_divergence_lognorm(shape, loc, scale, predicted_shape, predicted_loc, predicted_scale)
        w_dist = wasserstein_distance_lognorm(shape, loc, scale, predicted_shape, predicted_loc, predicted_scale)

        # Round values
        kl_div = round(kl_div, 3)
        w_dist = round(w_dist, 3)

        # Store the results in the dictionary
        results['Year'].append(year)
        results['KL Divergence'].append(kl_div)
        results['Wasserstein Distance'].append(w_dist)

    # Convert the results dictionary to a DataFrame for logging.infoing
    results_df = pd.DataFrame(results)
    logging.info(f'\n{results_df}')

def plot_comparisons(mort_df, predicted_shapes, predicted_locs, predicted_scales):
    for year in range(2011, 2023):
        mort_rates = mort_df[f'{year} MR'].values
        non_zero_mort_rates = mort_rates[mort_rates > 0]
        lognorm_params = lognorm.fit(non_zero_mort_rates)
        shape, loc, scale = lognorm_params
    
        predicted_shape = predicted_shapes[year]
        predicted_loc = predicted_locs[year]
        predicted_scale = predicted_scales[year]

        # Visual comparison of the distributions
        plt.figure(figsize=(10, 5))

        # Plot actual distribution
        x = np.linspace(non_zero_mort_rates.min(), non_zero_mort_rates.max(), 1000)
        plt.plot(x, lognorm.pdf(x, shape, loc, scale), label='Target Distribution', color='blue')

        # Plot predicted distribution
        plt.plot(x, lognorm.pdf(x, predicted_shape, predicted_loc, predicted_scale), label='Predicted Distribution', color='red', linestyle='dashed')

        plt.title(f'Distribution Comparison for Year {year}')
        plt.xlabel('MR')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'Autoencoder/Efficacy/Distribution Comparison/{year}_distribution_comparison.png')
        plt.close()

def print_top_errors(mort_df, preds_df, year=YEAR):
    err_df = mort_df[['FIPS']].copy()
    err_df[f'{year} Absolute Errors'] = abs(preds_df[f'{year} Preds'] - mort_df[f'{year} MR'])

    top_errors_df = err_df.sort_values(by=f'{year} Absolute Errors', ascending=False).head(10)
    top_errors_df_reset = top_errors_df.reset_index(drop=True)
    logging.info(f'\n{top_errors_df_reset}')

def main():
    mort_df = load_mortality(MORTALITY_PATH, MORTALITY_NAMES)
    # preds_df, predicted_shapes, predicted_locs, predicted_scales = load_predictions()
    preds_df = load_predictions()
    calculate_efficacy_metrics(mort_df, preds_df)
    # compare_distributions(mort_df, predicted_shapes, predicted_locs, predicted_scales)
    # plot_comparisons(mort_df, predicted_shapes, predicted_locs, predicted_scales)
    print_top_errors(mort_df, preds_df)

if __name__ == "__main__":
    main()