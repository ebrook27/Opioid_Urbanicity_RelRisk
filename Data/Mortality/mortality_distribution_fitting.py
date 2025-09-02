### 4/10/25, EB: Realized I was being naive with my relative risk binning. A more rigorous approach is to fit a distribution to the mortality data,
### and then I can look at the risk levels as percentiles of that distribution. This is a more robust way to define risk levels.

# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy import stats
# from scipy.stats import kstest

# # Load file
# df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
# df['FIPS'] = df['FIPS'].str.zfill(5)

# # Get all year columns
# year_cols = [col for col in df.columns if col.endswith('MR')]

# # Flatten all values into one big array (you can filter zeroes if needed)
# mortality_values = df[year_cols].values.flatten()
# mortality_values = mortality_values[~np.isnan(mortality_values)]
# mortality_values = mortality_values[mortality_values > 0]  # optional

# sns.histplot(mortality_values, kde=True, bins=50)
# plt.title("Histogram of All Mortality Rates (2010–2022)")
# plt.xlabel("Mortality Rate")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()

# # Fit a log-normal distribution
# lognorm_params = stats.lognorm.fit(mortality_values, floc=0)

# # Fit a gamma distribution
# gamma_params = stats.gamma.fit(mortality_values, floc=0)

# # Fit a Weibull distribution
# weibull_params = stats.weibull_min.fit(mortality_values, floc=0)

# x = np.linspace(min(mortality_values), max(mortality_values), 1000)

# plt.figure(figsize=(10, 6))
# sns.histplot(mortality_values, bins=50, stat='density', label='Empirical', alpha=0.4)

# # Overlay PDF curves
# plt.plot(x, stats.lognorm.pdf(x, *lognorm_params), label='Lognormal')
# plt.plot(x, stats.gamma.pdf(x, *gamma_params), label='Gamma')
# plt.plot(x, stats.weibull_min.pdf(x, *weibull_params), label='Weibull')

# plt.title("Fitted Distributions vs. Empirical Mortality Data")
# plt.xlabel("Mortality Rate")
# plt.ylabel("Density")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # # Define your custom bin edges using percentiles
# # cdf = stats.lognorm.cdf
# # inv_cdf = stats.lognorm.ppf  # inverse CDF = quantile function

# # # Example: create bins based on percentiles
# # cutoffs = inv_cdf(
# #     [0.005, 0.01, 0.02, 0.03, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 1.0], 
# #     *lognorm_params
# # )


# ### 4/14/25, EB: Now I'm trying to use AIC, BIC, and the Kolmogorov-Smirnov tests to determine which distribution fits best.
# ### I can then use the distribution to determine the percentiles for the risk levels.

# # Helper to compute AIC and log-likelihood
# def compute_metrics(data, dist_name, dist_obj, params):
#     loglik = np.sum(dist_obj.logpdf(data, *params))
#     k = len(params)
#     aic = 2 * k - 2 * loglik
#     bic = k * np.log(len(data)) - 2 * loglik
#     ks_stat, ks_p = kstest(data, dist_name, args=params)
#     return {
#         'Log-Likelihood': loglik,
#         'AIC': aic,
#         'BIC': bic,
#         'KS p-value': ks_p
#     }

# results = {
#     'Lognormal': compute_metrics(mortality_values, 'lognorm', stats.lognorm, lognorm_params),
#     'Gamma': compute_metrics(mortality_values, 'gamma', stats.gamma, gamma_params),
#     'Weibull': compute_metrics(mortality_values, 'weibull_min', stats.weibull_min, weibull_params),
# }

# results_df = pd.DataFrame(results).T
# print("📊 Distribution Fit Comparison:")
# print(results_df.sort_values('AIC'))







#################################################################################################################
#################################################################################################################

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load mortality data
df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
df['FIPS'] = df['FIPS'].str.zfill(5)
year_cols = [col for col in df.columns if col.endswith('MR')]

# Helper function to compute goodness-of-fit metrics
def compute_fit_metrics(data, dist_obj, dist_name, params):
    log_likelihood = np.sum(dist_obj.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(len(data)) - 2 * log_likelihood
    ks_stat, ks_p = stats.kstest(data, dist_name, args=params)
    return {
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'params': params
    }

# Store results
all_fit_results = []

# Loop over years
for col in year_cols:
    year = int(col.split()[0])
    values = df[col].dropna()
    values = values[values > 0].values  # filter zero mortality

    if len(values) == 0:
        continue

    # Fit distributions (force loc=0 for comparability)
    lognorm_params = stats.lognorm.fit(values)#, floc=0)
    gamma_params = stats.gamma.fit(values)#, floc=0)
    weibull_params = stats.weibull_min.fit(values)#, floc=0)

    # Compute metrics
    lognorm_fit = compute_fit_metrics(values, stats.lognorm, 'lognorm', lognorm_params)
    gamma_fit = compute_fit_metrics(values, stats.gamma, 'gamma', gamma_params)
    weibull_fit = compute_fit_metrics(values, stats.weibull_min, 'weibull_min', weibull_params)

    # Store in table format
    for dist_name, fit in zip(
        ['lognorm', 'gamma', 'weibull_min'],
        [lognorm_fit, gamma_fit, weibull_fit]
    ):
        all_fit_results.append({
            'Year': year,
            'Distribution': dist_name,
            'AIC': fit['aic'],
            'BIC': fit['bic'],
            'LogLikelihood': fit['log_likelihood'],
            'K-S Stat': fit['ks_stat'],
            'K-S p-value': fit['ks_p'],
            'Params': fit['params']
        })

    # Optional: Plot overlay
    # x = np.linspace(min(values), max(values), 1000)
    # plt.figure(figsize=(6, 4))
    # sns.histplot(values, bins=40, stat='density', alpha=0.4, label='Empirical')
    # plt.plot(x, stats.lognorm.pdf(x, *lognorm_params), label='Lognorm')
    # plt.plot(x, stats.gamma.pdf(x, *gamma_params), label='Gamma')
    # plt.plot(x, stats.weibull_min.pdf(x, *weibull_params), label='Weibull')
    # plt.title(f"Year {year} – Fitted Distributions")
    # plt.xlabel("Mortality Rate")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

# Compile results into a DataFrame
fit_df = pd.DataFrame(all_fit_results)

# # Display best model per year by AIC
# best_by_aic = fit_df.loc[fit_df.groupby("Year")["AIC"].idxmin()]
# print("✅ Best-fitting distribution per year (by AIC):")
# print(best_by_aic[["Year", "Distribution", "AIC", "BIC", "K-S p-value"]])

# Display best model per year by K-S p-value
best_by_ks = fit_df.loc[fit_df.groupby("Year")["K-S p-value"].idxmax()]
print("\n✅ Best-fitting distribution per year (by K-S p-value):")
print(best_by_ks[["Year", "Distribution", "K-S p-value", "K-S Stat", "AIC"]])


# Optionally save
#fit_df.to_csv("mortality_fit_by_year_all_distributions.csv", index=False)
