import pandas as pd
import numpy as np
from scipy.stats import lognorm, gamma, weibull_min, invgauss, kstest
import logging

# Constants
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]

# Set up logging
log_file = 'Log Files/distribution_fit_tests.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])


def load_mort_rates():
    mort_df = pd.read_csv(MORTALITY_PATH, header=0, names=MORTALITY_NAMES)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)  # Pad FIPS codes with leading zeros
    mort_df[MORTALITY_NAMES[1:]] = mort_df[MORTALITY_NAMES[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def test_goodness_of_fit(mort_df, year):
    """Tests goodness-of-fit for lognormal, gamma, and Weibull distributions using KS, AIC, and BIC."""

    # Get the mortality data for the selected year
    mort_rates = mort_df[f'{year} Mortality Rates'].values
    non_zero_mort_rates = mort_rates[mort_rates > 0]  # Ignore zero values for fitting

    # Function to calculate AIC and BIC
    def calculate_aic_bic(log_likelihood, num_params, n):
        aic = 2 * num_params - 2 * log_likelihood
        bic = num_params * np.log(n) - 2 * log_likelihood
        return aic, bic

    # Initialize results dictionary
    results = {}

    # Lognormal distribution
    log_shape, log_loc, log_scale = lognorm.fit(non_zero_mort_rates)
    log_likelihood = np.sum(lognorm.logpdf(non_zero_mort_rates, log_shape, loc=log_loc, scale=log_scale))
    ks_stat, ks_p = kstest(non_zero_mort_rates, 'lognorm', args=(log_shape, log_loc, log_scale))
    aic, bic = calculate_aic_bic(log_likelihood, 3, len(non_zero_mort_rates))
    results['Lognormal'] = {'KS Stat': ks_stat, 'p-value': ks_p, 'AIC': aic, 'BIC': bic}

    # Gamma distribution
    gamma_shape, gamma_loc, gamma_scale = gamma.fit(non_zero_mort_rates, floc=0)
    log_likelihood = np.sum(gamma.logpdf(non_zero_mort_rates, gamma_shape, loc=gamma_loc, scale=gamma_scale))
    ks_stat, ks_p = kstest(non_zero_mort_rates, 'gamma', args=(gamma_shape, gamma_loc, gamma_scale))
    aic, bic = calculate_aic_bic(log_likelihood, 3, len(non_zero_mort_rates))
    results['Gamma'] = {'KS Stat': ks_stat, 'p-value': ks_p, 'AIC': aic, 'BIC': bic}

    # Weibull distribution
    weibull_shape, weibull_loc, weibull_scale = weibull_min.fit(non_zero_mort_rates, floc=0)
    log_likelihood = np.sum(weibull_min.logpdf(non_zero_mort_rates, weibull_shape, loc=weibull_loc, scale=weibull_scale))
    ks_stat, ks_p = kstest(non_zero_mort_rates, 'weibull_min', args=(weibull_shape, weibull_loc, weibull_scale))
    aic, bic = calculate_aic_bic(log_likelihood, 3, len(non_zero_mort_rates))
    results['Weibull'] = {'KS Stat': ks_stat, 'p-value': ks_p, 'AIC': aic, 'BIC': bic}

    # Inverse Gaussian distribution
    invgauss_shape, invgauss_loc, invgauss_scale = invgauss.fit(non_zero_mort_rates, floc=0)
    log_likelihood = np.sum(invgauss.logpdf(non_zero_mort_rates, invgauss_shape, loc=invgauss_loc, scale=invgauss_scale))
    ks_stat, ks_p = kstest(non_zero_mort_rates, 'invgauss', args=(invgauss_shape, invgauss_loc, invgauss_scale))
    aic, bic = calculate_aic_bic(log_likelihood, 3, len(non_zero_mort_rates))
    results['Inverse Gaussian'] = {'KS Stat': ks_stat, 'p-value': ks_p, 'AIC': aic, 'BIC': bic}

    # Print results
    logging.info(f"Year {year} - Goodness-of-Fit Tests:")
    for dist, metrics in results.items():
        logging.info(f"  {dist} Distribution:")
        logging.info(f"    KS Stat = {metrics['KS Stat']:.4f}, p-value = {metrics['p-value']:.4f}")
        logging.info(f"    AIC = {metrics['AIC']:.2f}, BIC = {metrics['BIC']:.2f}\n")

def main():
    mort_df = load_mort_rates()

    # Test distributions for each year
    for year in range(2010, 2023):
        test_goodness_of_fit(mort_df, year)

if __name__ == "__main__":
    main()
