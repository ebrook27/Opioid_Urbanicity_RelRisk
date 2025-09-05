### 9/4/25, EB: Trying to fit a Tweedie distribution to the yearly mortality data. Might account for zero-inflation.

import numpy as np
import pandas as pd
#from sklearn.linear_model import TweedieRegressor

mortality_rates = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv')

# regressor_types = {'Normal': 0, 'Poisson': 1, 'Compound Poisson Gamma': 1.5, 'Gamma': 2, 'Inverse Gaussian': 3}

# for year in mortality_rates['Year']:
#     for dist, power in regressor_types:
#         reg = TweedieRegressor(power=power, alpha=0.5, link='log')
#         reg.fit(mortality_rates[year])


from scipy.stats import tweedie

data = mortality_rates["MR 2020"].values

# Fit Tweedie distribution parameters (requires specifying p)
p = 1.5
mean, var, skew, kurt = tweedie.stats(p, mu=data.mean(), phi=data.var(), moments='mvsk')
print(mean, var, skew, kurt)





############################################
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from statsmodels.genmod.families.family import Tweedie
# from scipy.optimize import minimize

# # -----------------------------
# # Helper: Fit Tweedie for fixed p
# # -----------------------------
# def fit_tweedie(data, p):
#     """Fit Tweedie distribution parameters (mu, phi) by MLE for fixed p."""
#     data = np.asarray(data)
#     data = data[~np.isnan(data)]  # drop NaNs

#     def nll(params):
#         mu, phi = params
#         if mu <= 0 or phi <= 0:
#             return np.inf
#         dist = Tweedie(var_power=p, mean=mu, dispersion=phi)
#         return -np.sum(dist.logpdf(data))

#     init_params = [np.mean(data), np.var(data) / (np.mean(data) ** p)]
#     res = minimize(
#         nll, init_params, method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)]
#     )
#     mu_hat, phi_hat = res.x
#     return mu_hat, phi_hat, -res.fun  # return log-likelihood too


# # -----------------------------
# # Fit across candidate p values
# # -----------------------------
# def best_tweedie_fit(data, p_grid=np.linspace(1.1, 1.9, 9)):
#     best = None
#     for p in p_grid:
#         try:
#             mu, phi, ll = fit_tweedie(data, p)
#             if (best is None) or (ll > best["ll"]):
#                 best = {"p": p, "mu": mu, "phi": phi, "ll": ll}
#         except Exception:
#             continue
#     return best

# def tweedie_zero_prob(mu, phi, p):
#     """Return Tweedie probability mass at zero (valid for 1<p<2)."""
#     if p <= 1 or p >= 2:
#         return 0.0
#     return np.exp(- (mu ** (2 - p)) / (phi * (2 - p)))


# def plot_yearly_fits(df, p_grid=np.linspace(1.1, 1.9, 9)):
#     year_cols = [c for c in df.columns if c.startswith("MR")]
#     results = []

#     for col in year_cols:
#         data = df[col].dropna().values
#         fit = best_tweedie_fit(data, p_grid)

#         if fit is None:
#             print(f"⚠️ Could not fit Tweedie for {col} (data may be all zeros or optimization failed)")
#             continue

#         results.append({"year": col, **fit})
#         dist = Tweedie(var_power=fit["p"], mean=fit["mu"], dispersion=fit["phi"])

#         # Histogram
#         plt.figure(figsize=(6, 4))
#         plt.hist(data[data > 0], bins=30, density=True, alpha=0.6,
#                  color="skyblue", label="Observed (nonzero)")

#         # PDF overlay
#         x = np.linspace(data[data > 0].min(), data.max(), 300)
#         pdf_vals = dist.pdf(x)
#         plt.plot(x, pdf_vals, "r-", lw=2,
#                  label=f"Tweedie fit (p={fit['p']:.2f})")

#         # Zero mass: observed vs fitted
#         obs_zero_frac = np.mean(data == 0)
#         fit_zero_prob = tweedie_zero_prob(fit["mu"], fit["phi"], fit["p"])

#         plt.bar(0, obs_zero_frac, width=0.1, color="gray", alpha=0.6,
#                 label=f"Observed zeros ({obs_zero_frac:.2f})")
#         plt.bar(0.2, fit_zero_prob, width=0.1, color="red", alpha=0.6,
#                 label=f"Fitted zero mass ({fit_zero_prob:.2f})")

#         plt.title(f"{col} — Mortality Rates")
#         plt.xlabel("Mortality rate")
#         plt.ylabel("Density / Probability")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     return pd.DataFrame(results)



# # -----------------------------
# # Example usage
# # -----------------------------
# fit_results = plot_yearly_fits(mortality_rates)
# print(fit_results)