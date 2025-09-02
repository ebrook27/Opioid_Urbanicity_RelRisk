import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gamma, weibull_min, expon, invgauss, pareto, kstest

# Constants
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]
TAIL = 3  # Tails for anomaly detection

def load_mort_rates():
    mort_df = pd.read_csv(MORTALITY_PATH, header=0, names=MORTALITY_NAMES)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)  # Pad FIPS codes with leading zeros
    mort_df[MORTALITY_NAMES[1:]] = mort_df[MORTALITY_NAMES[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def count_zero_values(mort_df, year):
    zero_count = (mort_df[f'{year} Mortality Rates'] == 0).sum()
    print(f'Year {year}: {zero_count} counties have zero mortality rates.')

def fit_distribution(mort_df, year):
    # Get the mortality data for the selected year
    mort_rates = mort_df[f'{year} Mortality Rates'].values
    non_zero_mort_rates = mort_rates[mort_rates > 0]  # Ignore zero values for fitting
    
    # Fit the lognormal distribution
    log_shape, log_loc, log_scale = lognorm.fit(non_zero_mort_rates)
    
    # Perform K-S test for lognormal distribution
    ks_stat, p_value = kstest(non_zero_mort_rates, 'lognorm', args=(log_shape, log_loc, log_scale))
    print(f"Year {year} - Lognormal K-S Test: KS Stat={ks_stat:.4f}, p-value={p_value:.4f}\n")
    
    # Fit the gamma distribution
    gamma_shape, gamma_loc, gamma_scale = gamma.fit(non_zero_mort_rates, floc=0)
    
    # Fit the Weibull distribution
    weibull_shape, weibull_loc, weibull_scale = weibull_min.fit(non_zero_mort_rates, floc=0)

    # Fit the exponential distribution
    exp_loc, exp_scale = expon.fit(non_zero_mort_rates, floc=0)

    # Fit the inverse Gaussian (Wald) distribution
    invgauss_shape, invgauss_loc, invgauss_scale = invgauss.fit(non_zero_mort_rates, floc=0)

    # Fit the Pareto distribution
    pareto_shape, pareto_loc, pareto_scale = pareto.fit(non_zero_mort_rates, floc=0)

    # Generate points for the fitted distributions
    x_vals = np.linspace(non_zero_mort_rates.min(), non_zero_mort_rates.max(), 1000)
    
    # PDFs
    lognormal_pdf = lognorm.pdf(x_vals, log_shape, loc=log_loc, scale=log_scale)
    gamma_pdf = gamma.pdf(x_vals, gamma_shape, loc=gamma_loc, scale=gamma_scale)
    weibull_pdf = weibull_min.pdf(x_vals, weibull_shape, loc=weibull_loc, scale=weibull_scale)
    exp_pdf = expon.pdf(x_vals, loc=exp_loc, scale=exp_scale)
    invgauss_pdf = invgauss.pdf(x_vals, invgauss_shape, loc=invgauss_loc, scale=invgauss_scale)
    pareto_pdf = pareto.pdf(x_vals, pareto_shape, loc=pareto_loc, scale=pareto_scale)

    # Plot the histogram of the actual data
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_mort_rates, bins=30, density=True, alpha=0.6, color='b', label='Mortality Data')
    
    # Plot the fitted distributions
    plt.plot(x_vals, lognormal_pdf, 'r-', lw=2, label=f'Fitted Lognormal')
    plt.plot(x_vals, gamma_pdf, 'g-', lw=2, label=f'Fitted Gamma')
    plt.plot(x_vals, weibull_pdf, 'm-', lw=2, label=f'Fitted Weibull')
    plt.plot(x_vals, exp_pdf, 'c-', lw=2, label=f'Fitted Exponential')
    plt.plot(x_vals, invgauss_pdf, 'y-', lw=2, label=f'Fitted Inverse Gaussian')

    # Terrible fits (uncomment to see how bad)
    # plt.plot(x_vals, pareto_pdf, 'k-', lw=2, label=f'Fitted Pareto') 

    # Add labels and title
    plt.title(f'{year} Mortality Rates and Fitted Distributions', size=16)
    plt.xlabel('Mortality Rate', size=14)
    plt.ylabel('Density', size=14)
    
    # Add a legend
    plt.legend(loc='upper right')
    
    plt.show()
    
    # Save the plot
    #plt.savefig(f'Anomalies/Fitted Distributions/{year}_fitted_distributions.png')

def main():
    mort_df = load_mort_rates()

    for year in range(2010, 2023):
        count_zero_values(mort_df, year)
        fit_distribution(mort_df, year)

if __name__ == "__main__":
    main()