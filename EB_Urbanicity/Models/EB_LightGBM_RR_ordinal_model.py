### 4/28/25, EB: I tried using an Logistic ordinal regression model to predict the relative risk levels in RR_top20_classification.py. 
### It worked ok, but mostly just predicted counties into the lowest risk category, which has the most counties. So to try to address this imbalance,
### I am trying a LightGBM ordinal regression model. First we will use the LightGBM to predict mortality rates, and then bin the outputs into the ordinal risk levels.
### Once we have that set up correctly, I will try to build a custom loss function to directly predict the ordinal risk levels.

from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, mean_absolute_error)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
### Troubleshooting: To avoid an error in mord, we need to set the deprecated np.int to the default int type.
np.int = int
import warnings
# from collections import Counter
# from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings("ignore", message=".*use_inf_as_na option is deprecated.*")
from scipy.stats import lognorm
from functools import reduce
from mord import LogisticAT
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from scipy import sparse



mortality_path = 'Data\Mortality\Final Files\Mortality_final_rates.csv'
log_path = 'County Classification\RR_top20_LightGBM_ordinal_classification_log.txt'

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']


# Your original desired percentiles
PERCENTILES = [99.9, 99.5, 99, 95, 90, 85, 80]

def adaptive_log_normal_binning(mortality_series, min_bin_size=10, step_size=0.5):
    """
    Performs adaptive binning using a fitted log-normal distribution.
    Starts with fixed percentiles for each bin, and adaptively lowers cutoff
    if there are too few counties. Resets to the next target percentile for each bin.
    
    Parameters:
        mortality_series (pd.Series): Mortality rates indexed by FIPS
        min_bin_size (int): Minimum number of counties per bin
        step_size (float): Step size (percent) to lower cutoff if bin is too small

    Returns:
        bin_assignments (dict): Mapping from FIPS to bin index
        final_cutoffs (list): Final adjusted percentile cutoffs used
    """
    # Fit log-normal to mortality data
    values = mortality_series.dropna()
    values = values[values > 0]  # Avoid zeros
    shape, loc, scale = lognorm.fit(values, floc=0)
    
    bin_assignments = {}
    assigned_fips = set()
    final_cutoffs = []
    
    # Loop through intended percentiles
    for bin_idx, target_percentile in enumerate(PERCENTILES[:-1]):  # Ignore the last 80% for now
        current_percentile = target_percentile
        found = False
        
        while current_percentile > PERCENTILES[-1]:  # Don't drop below 80th percentile
            cutoff_value = lognorm.ppf(current_percentile/100, shape, loc=loc, scale=scale)
            
            # Eligible counties for this bin
            eligible = mortality_series[
                (mortality_series > cutoff_value) &
                (~mortality_series.index.isin(assigned_fips))
            ]
            
            if len(eligible) >= min_bin_size:
                # Accept this bin
                for fips in eligible.index:
                    bin_assignments[fips] = bin_idx
                assigned_fips.update(eligible.index)
                final_cutoffs.append(current_percentile)
                found = True
                break  # Move to next bin
            
            current_percentile -= step_size  # Lower threshold and retry
        
        if not found:
            # No acceptable bin found, still record the last tried cutoff
            final_cutoffs.append(current_percentile)
    
    # Remaining counties above 80% cutoff assigned to last bin
    cutoff_value = lognorm.ppf(PERCENTILES[-1]/100, shape, loc=loc, scale=scale)
    remaining = mortality_series[
        (mortality_series > cutoff_value) &
        (~mortality_series.index.isin(assigned_fips))
    ]
    for fips in remaining.index:
        bin_assignments[fips] = len(PERCENTILES) - 2  # Last bin index (note: PERCENTILES[:-1] above)
    final_cutoffs.append(PERCENTILES[-1])

    return bin_assignments, final_cutoffs


def prepare_yearly_prediction_data_adaptive(mortality_path):
    """Creates a long-format dataset for predicting next-year mortality using current-year SVI + county class."""
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2022))  # We predict mortality up to 2022

    # Load county category
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

    # Load raw mortality
    mortality_df = pd.read_csv(mortality_path, dtype={'FIPS': str})
    mortality_df['FIPS'] = mortality_df['FIPS'].str.zfill(5)
    mortality_df = mortality_df.set_index('FIPS')

    # Load SVI data
    svi_data = []
    for var in svi_variables:
        var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
        var_df = pd.read_csv(var_path, dtype={'FIPS': str})
        var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
        long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)
        long_df['year'] = long_df['year_var'].str.extract(r'(\d{4})').astype(int)
        long_df = long_df[long_df['year'].between(2010, 2021)]
        long_df = long_df.drop(columns='year_var')
        svi_data.append(long_df)

    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Add mortality bins based on fitted log-normal
    rr_bins = []
    for y in years:
        target_year = y + 1
        mort_col = f"{target_year} MR"
        if mort_col not in mortality_df.columns:
            continue

        year_df = svi_merged[svi_merged['year'] == y].copy()
        mortality_year = mortality_df[mort_col]

        # Adaptive binning
        bin_assignments, final_cutoffs = adaptive_log_normal_binning(mortality_year, min_bin_size=10, step_size=0.5)

        # Map bins
        year_df['mortality_next'] = year_df['FIPS'].map(mortality_year)
        year_df['rr_bin'] = year_df['FIPS'].map(bin_assignments)

        # Save processed year
        rr_bins.append(year_df)
        
        # Optional: Print final adjusted cutoffs for diagnostics
        print(f"Year {y}: Final bin cutoffs = {final_cutoffs}")

    all_years = pd.concat(rr_bins, ignore_index=True)

    # Add county class
    all_years = all_years.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows outside top 20% or with missing values
    all_years = all_years.dropna(subset=['rr_bin'] + svi_variables + ['county_class'])

    # Convert to integer
    all_years['rr_bin'] = all_years['rr_bin'].astype(int)

    return all_years


# def yearly_regression_prediction_binning(df, n_splits=5, log_path='lgbm_ordinal_log.txt'):
#     """
#     Predicts mortality rates using LightGBM regressor, bins predictions using adaptive log-normal fit,
#     and evaluates as an ordinal classification problem.
#     """
#     metrics_all_years = []

#     with open(log_path, 'w') as log:
#         for year in range(2010, 2021):
#             print(f"🔁 Processing year {year} → predicting RR bin for year {year+1}")
#             df_year = df[df['year'] == year].copy()

#             if df_year.empty:
#                 print(f"⚠️ Skipping year {year}: no data.")
#                 continue

#             feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_next', 'rr_bin']]
#             X = df_year[feature_cols]
#             y_continuous = df_year['mortality_next']

#             skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#             fold_metrics = []

#             # Important: Bin y_continuous to stratify correctly
#             y_bins_for_stratify = df_year['rr_bin'].astype(int)

#             for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_bins_for_stratify)):
#                 X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#                 y_train_continuous = y_continuous.iloc[train_idx]
#                 y_test_continuous = y_continuous.iloc[test_idx]
#                 y_test_true_bins = y_bins_for_stratify.iloc[test_idx]

#                 # === Preprocessing + Model Pipeline ===
#                 preprocessor = ColumnTransformer([
#                     ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['county_class'])
#                 ], remainder='passthrough')

#                 pipeline = Pipeline([
#                     ('prep', preprocessor),
#                     ('model', LGBMRegressor(
#                         n_estimators=500,
#                         learning_rate=0.05,
#                         max_depth=7,
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         random_state=42,
#                         n_jobs=-1
#                     ))
#                 ])

#                 pipeline.fit(X_train, y_train_continuous)
#                 y_pred_continuous = pipeline.predict(X_test)

#                 # === Binning Step ===
#                 positive_train = y_train_continuous[y_train_continuous > 0]
#                 shape, loc, scale = lognorm.fit(positive_train, floc=0)

#                 # Generate cutoffs
#                 percentiles = [97, 95, 93, 90, 85, 80]
#                 cutoffs = [lognorm.ppf(p/100, shape, loc=loc, scale=scale) for p in percentiles]

#                 # Bin predictions
#                 y_pred_bins = np.digitize(y_pred_continuous, cutoffs)
#                 # Bin true test values
#                 y_test_bins = np.digitize(y_test_continuous, cutoffs)

#                 # === Metrics ===
#                 acc = accuracy_score(y_test_bins, y_pred_bins)
#                 f1 = f1_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0)
#                 recall = recall_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0)
#                 precision = precision_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0)
#                 mae_continuous = mean_absolute_error(y_test_continuous, y_pred_continuous)

#                 fold_metrics.append({
#                     'fold': fold_idx + 1,
#                     'Accuracy': acc,
#                     'F1_weighted': f1,
#                     'Recall_weighted': recall,
#                     'Precision_weighted': precision,
#                     'MAE_continuous': mae_continuous
#                 })

#                 # === Logging for each fold ===
#                 log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
#                 log.write(f"Cutoffs used: {cutoffs}\n")
#                 log.write(classification_report(y_test_bins, y_pred_bins, digits=3))
#                 log.write("\nConfusion Matrix:\n")

#                 labels_present = sorted(np.unique(y_test_bins))
#                 conf_mat = confusion_matrix(y_test_bins, y_pred_bins, labels=labels_present)
#                 conf_df = pd.DataFrame(
#                     conf_mat,
#                     index=[f'True {i}' for i in labels_present],
#                     columns=[f'Pred {i}' for i in labels_present]
#                 )
#                 log.write(conf_df.to_string())
#                 log.write("\n\n")

#             fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
#             year_metrics = fold_df.mean().to_dict()
#             year_metrics['Year'] = year
#             metrics_all_years.append(year_metrics)

#     metrics_df = pd.DataFrame(metrics_all_years)
#     return metrics_df



# def plot_results_lightgbm(metrics_df):
#     """
#     Plot accuracy, F1 (weighted), recall (weighted), precision (weighted), and continuous MAE over prediction years.
#     """
#     plt.figure(figsize=(10, 6))
    
#     sns.lineplot(data=metrics_df, x='Year', y='Accuracy', label='Accuracy', marker='o')
#     sns.lineplot(data=metrics_df, x='Year', y='F1_weighted', label='F1 Score (Weighted)', marker='s')
#     sns.lineplot(data=metrics_df, x='Year', y='Recall_weighted', label='Recall (Weighted)', marker='^')
#     sns.lineplot(data=metrics_df, x='Year', y='Precision_weighted', label='Precision (Weighted)', marker='x')

#     # Plot MAE separately with secondary y-axis
#     ax1 = plt.gca()
#     ax2 = ax1.twinx()
#     sns.lineplot(data=metrics_df, x='Year', y='MAE_continuous', label='Mean Absolute Error', marker='D', color='k', ax=ax2)

#     ax1.set_title("LightGBM Regression + Binned Classification Performance Over Time")
#     ax1.set_xlabel("Prediction Year")
#     ax1.set_ylabel("Classification Metrics (0–1 scale)")
#     ax2.set_ylabel("Continuous Mortality MAE")
    
#     ax1.set_ylim(0, 1)
#     ax1.grid(True, linestyle='--', alpha=0.5)
    
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

#     plt.tight_layout()
#     plt.show()

def plot_results_lightgbm(metrics_df):
    """
    Plot classification metrics and continuous MAE over prediction years in two side-by-side plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    # === Left Plot: Classification Metrics ===
    sns.lineplot(data=metrics_df, x='Year', y='Accuracy', label='Accuracy', marker='o', ax=ax1)
    sns.lineplot(data=metrics_df, x='Year', y='F1_weighted', label='F1 Score (Weighted)', marker='s', ax=ax1)
    sns.lineplot(data=metrics_df, x='Year', y='Recall_weighted', label='Recall (Weighted)', marker='^', ax=ax1)
    sns.lineplot(data=metrics_df, x='Year', y='Precision_weighted', label='Precision (Weighted)', marker='x', ax=ax1)

    ax1.set_title("Classification Metrics Over Time")
    ax1.set_xlabel("Prediction Year")
    ax1.set_ylabel("Score (0–1)")
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # === Right Plot: Continuous Mortality MAE ===
    sns.lineplot(data=metrics_df, x='Year', y='MAE_continuous', marker='D', color='red', ax=ax2)

    ax2.set_title("Continuous Mortality Prediction MAE Over Time")
    ax2.set_xlabel("Prediction Year")
    ax2.set_ylabel("Mean Absolute Error")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


#### 4/28/25, EB: The above are the functions that are used to predict mortaltiy rates and then bin the results into ordinal risk levels.
#### Now I'm trying to use a custom loss function to directly predict the ordinal risk levels.
#### I will use the LightGBM ordinal regression model to do this.

from scipy.special import expit  # Sigmoid function

def custom_ordinal_loss(y_true, y_pred):
    """
    Custom ordinal regression loss for LightGBM (cumulative logistic model).
    
    Args:
        y_true: shape (n_samples,)
        y_pred: shape (n_samples * (n_bins - 1),)
    Returns:
        grad: gradient
        hess: hessian
    """
    n_classes = 6  # Your case: bins 0–5
    n_thresholds = n_classes - 1
    n_samples = y_true.shape[0]

    y_pred = y_pred.reshape(n_thresholds, n_samples).T  # Reshape to (n_samples, n_thresholds)

    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)

    for k in range(n_thresholds):
        p = expit(y_pred[:, k])
        grad[:, k] = p - (y_true <= k).astype(int)
        hess[:, k] = p * (1 - p)

    return grad.flatten(), hess.flatten()

def predict_ordinal(y_pred_raw, n_classes):
    n_thresholds = n_classes - 1
    n_samples = len(y_pred_raw) // n_thresholds
    preds = y_pred_raw.reshape(n_samples, n_thresholds)

    prob = expit(preds)  # Sigmoid
    cum_prob = np.cumprod(prob, axis=1)

    final_preds = (cum_prob > 0.5).sum(axis=1)
    return final_preds

### 4/28/25, EB: Here is a custom training loop that enables LightGBM to use the custom loss function above.


def yearly_ordinal_prediction_custom(df, n_splits=5, log_path='lgbm_customordinal_log.txt', n_classes=6):
    """
    Predicts ordinal risk bins using LightGBM with a custom ordinal loss.
    """
    metrics_all_years = []

    with open(log_path, 'w') as log:
        for year in range(2010, 2021):
            print(f"🔁 Processing year {year} → predicting RR bin for year {year+1}")
            df_year = df[df['year'] == year].copy()

            if df_year.empty:
                print(f"⚠️ Skipping year {year}: no data.")
                continue

            feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_next', 'rr_bin']]
            X = df_year[feature_cols]
            y_bins_for_stratify = df_year['rr_bin'].astype(int)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_bins_for_stratify)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train = y_bins_for_stratify.iloc[train_idx]
                y_test = y_bins_for_stratify.iloc[test_idx]

                # === Preprocessing ===
                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['county_class'])
                ], remainder='passthrough')

                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)

                # === LightGBM Dataset ===
                train_data = lgb.Dataset(X_train_processed, label=y_train)
                valid_data = lgb.Dataset(X_test_processed, label=y_test)#, reference=train_data)

                # params = {
                #     'objective': 'none',  # We override with custom loss
                #     'boosting_type': 'gbdt',
                #     'metric': 'None',
                #     'learning_rate': 0.05,
                #     'max_depth': 7,
                #     'num_leaves': 31,
                #     'subsample': 0.8,
                #     'colsample_bytree': 0.8,
                #     'n_jobs': -1,
                #     'verbose': -1,
                #     'seed': 42
                # }

                # model = lgb.train(
                #     params,
                #     train_set=train_data,
                #     num_boost_round=500,
                #     valid_sets=[valid_data],
                #     fobj=custom_ordinal_loss,
                #     verbose_eval=False
                # )
                
                params = {
                    'boosting_type': 'gbdt',
                    'objective': custom_ordinal_loss,  # Pass your custom function here
                    'metric': 'None',
                    'learning_rate': 0.05,
                    'max_depth': 7,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_jobs': -1,
                    'verbose': -1,
                    'seed': 42
                }

                model = lgb.train(
                    params,
                    train_set=train_data,
                    num_boost_round=500,
                    valid_sets=[valid_data]#,
                    #verbose_eval=False
                )


                # === Prediction ===
                y_pred_raw = model.predict(X_test_processed)
                y_pred_bins = predict_ordinal(y_pred_raw, n_classes=n_classes)

                # === Metrics ===
                acc = accuracy_score(y_test, y_pred_bins)
                f1 = f1_score(y_test, y_pred_bins, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred_bins, average='weighted', zero_division=0)
                precision = precision_score(y_test, y_pred_bins, average='weighted', zero_division=0)

                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'Accuracy': acc,
                    'F1_weighted': f1,
                    'Recall_weighted': recall,
                    'Precision_weighted': precision
                })

                # === Logging for each fold ===
                log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
                log.write(classification_report(y_test, y_pred_bins, digits=3))
                log.write("\nConfusion Matrix:\n")

                labels_present = sorted(np.unique(y_test))
                conf_mat = confusion_matrix(y_test, y_pred_bins, labels=labels_present)
                conf_df = pd.DataFrame(
                    conf_mat,
                    index=[f'True {i}' for i in labels_present],
                    columns=[f'Pred {i}' for i in labels_present]
                )
                log.write(conf_df.to_string())
                log.write("\n\n")

            fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
            year_metrics = fold_df.mean().to_dict()
            year_metrics['Year'] = year
            metrics_all_years.append(year_metrics)

    metrics_df = pd.DataFrame(metrics_all_years)
    return metrics_df
 



def main():
    print("Starting top 20% Ordinal regression classification and logging...")
    all_years_df = prepare_yearly_prediction_data_adaptive(mortality_path)
#    print(all_years_df.groupby(['year', 'rr_bin']).size().unstack(fill_value=0))
    metrics_df = yearly_ordinal_prediction_custom(all_years_df, n_splits=5, log_path=log_path)
    plot_results_lightgbm(metrics_df)
    print("✅ Ordinal regression classification and logging complete!")



if __name__ == "__main__":
    main()
