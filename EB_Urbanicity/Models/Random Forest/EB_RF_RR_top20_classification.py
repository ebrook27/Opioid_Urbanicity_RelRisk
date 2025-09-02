### 4/24/25, EB: After fighting with different ways to approach the prediction, I talked with Andrew and decided to try returning to the classification model, but focusing on the top 20% of mortality/relative risk.
### We can then split this top 20% into the risk levels we're ultimately interested in, the top 0.1%, top 0.5%, top 1%, and so on.
### This turns it from a huge 20-category classification problem into a more manageable 7 category classification problem.


from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, mean_absolute_error
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
from mord import LogisticAT, LogisticIT, OrdinalRidge
from sklearn.model_selection import GridSearchCV


mortality_path = 'Data\Mortality\Final Files\Mortality_final_rates.csv'
log_path = 'County Classification\RR_top20_classification_log.txt'

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

#PERCENTILES = [95, 90, 85, 80, 75]#99.5, 99.9, 99.5, 99, 95, 90, 85, 80]  # Percentiles for RR bins


### The next three lines are for when I tried using rank-ordered bins, rather than log-normal bins.
RANK_PERCENTILES = [0.01, 0.05, 0.10, 0.15, 0.20]#0.001,0.005, 

def assign_bin_by_rank(mortality_series):
    """
    Assigns counties to risk bins based on rank (not log-normal threshold).
    Only the top 20% of counties (by mortality) are assigned to a bin.
    """
    n_total = len(mortality_series)
    n_top20 = int(np.floor(n_total * 0.20))
    
    # Sort by mortality
    sorted_fips = mortality_series.sort_values(ascending=False).head(n_top20)
    
    # Determine bin thresholds based on rank
    bin_edges = [int(np.ceil(n_top20 * p)) for p in RANK_PERCENTILES]

    # Assign bins
    rank_bins = {}
    for i, (start, end) in enumerate(zip([0] + bin_edges[:-1], bin_edges)):
        fips_in_bin = sorted_fips.iloc[start:end].index
        for f in fips_in_bin:
            rank_bins[f] = i  # Bin 0 to 6

    return rank_bins

def prepare_yearly_prediction_data_rankorder(mortality_path):
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

        # Get bin assignments by rank
        bin_assignments = assign_bin_by_rank(mortality_year)

        # Apply bin assignments
        year_df['mortality_next'] = year_df['FIPS'].map(mortality_year)
        year_df['rr_bin'] = year_df['FIPS'].map(bin_assignments)

        rr_bins.append(year_df)

    all_years = pd.concat(rr_bins, ignore_index=True)

    # Add county class
    all_years = all_years.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows outside top 20% or with missing values
    all_years = all_years.dropna(subset=['rr_bin'] + svi_variables + ['county_class'])

    # Convert to integer
    all_years['rr_bin'] = all_years['rr_bin'].astype(int)

    return all_years


### This is the base constructor function, which uses the log-normal distribution to assign bins, using the percentiles from above.
def prepare_yearly_prediction_data_lognormal(mortality_path):
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
        year_mortality = mortality_df[mort_col].dropna()

        # Fit log-normal (use log of values for fitting mu/sigma)
        #log_values = np.log(year_mortality[year_mortality > 0])  # avoid zeros
        shape, loc, scale = lognorm.fit(year_mortality[year_mortality > 0], floc=0)

        # Compute thresholds using percentiles
        thresholds = [lognorm.ppf(p/100, shape, loc=loc, scale=scale) for p in PERCENTILES]

        def assign_bin(mortality_value):
            if pd.isna(mortality_value) or mortality_value <= thresholds[-1]:
                return np.nan  # Below top 20%, drop later
            for i, thresh in enumerate(thresholds):
                if mortality_value > thresh:
                    return i  # Lower i = higher risk
            return len(thresholds)  # fallback
                #     if i == 0:
                #         return 0  # Combine top 0.1% and 0.5%
                #     else:
                #         return i - 1  # Shift bins down by 1
                # return len(thresholds) - 1  # Fallback: lowest included bin

        year_df['mortality_next'] = year_df['FIPS'].map(mortality_df[mort_col])
        year_df['rr_bin'] = year_df['mortality_next'].apply(assign_bin)

        rr_bins.append(year_df)

    all_years = pd.concat(rr_bins, ignore_index=True)

    # Add county class
    all_years = all_years.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows outside top 20% or with missing values
    all_years = all_years.dropna(subset=['rr_bin'] + svi_variables + ['county_class'])

    # Convert to integer
    all_years['rr_bin'] = all_years['rr_bin'].astype(int)

    return all_years

###4/25/25, EB: Here I am trying to use an adaptive approach to the binning of the mortality data.
### When simply using set percentile cutoffs, I kept getting bins that had 5 or fewer counties in them. This is a problem, since we need to do k-fold CV on the data,
### to get predictions for all the counties. If we have bins with 5 or fewer counties, then we can't do k-fold CV on them.
### So here I am checking how many counties end up in each bin, and if there are fewer than 5 counties in a bin, I am moving dropping the percentile cutoff down by 0.5%.

import numpy as np
import pandas as pd
from scipy.stats import lognorm

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


#################################################################
### The following functions are for the classification model. ###


def yearly_classification_prediction_logging(df, n_splits=5, log_path=log_path):
    """
    Classifies counties into log-normal-based RR bins (0–6) for each year using k-fold cross-validation.
    Logs detailed performance metrics and confusion matrices for every fold.
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
            y_class = df_year['rr_bin'].astype(int)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(drop='first'), ['county_class'])
                ], remainder='passthrough')

                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('model', RandomForestClassifier(
                        n_estimators=250,
                        max_depth=15,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1
                    ))
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'Accuracy': acc,
                    'F1_macro': f1,
                    'Recall_macro': recall,
                    'Precision_macro': precision
                })

                # === Logging for each fold ===
                log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
                log.write(classification_report(y_test, y_pred, digits=3))
                log.write("\nConfusion Matrix:\n")

                labels_present = sorted(y_class.unique())
                conf_mat = confusion_matrix(y_test, y_pred, labels=labels_present)
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


def yearly_classification_prediction_logging_AUC(df, n_splits=5, log_path=log_path):
    """
    Classifies counties into log-normal-based RR bins using k-fold cross-validation.
    Logs performance reports and computes macro-AUC (OvR) using predicted probabilities.
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
            y_class = df_year['rr_bin'].astype(int)
            
            ### Troubleshooting imbalanced classes
            class_counts = y_class.value_counts()
            if class_counts.min() < 2:
                print(f"⚠️ Skipping year {year}: class with <2 samples present.")
                continue

            class_labels = sorted(y_class.unique())  # Needed for binarization & labels
            y_binarized = label_binarize(y_class, classes=class_labels)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['county_class'])
                ], remainder='passthrough')

                # pipeline = Pipeline([
                #     ('prep', preprocessor),
                #     ('model', RandomForestClassifier(
                #         n_estimators=250,
                #         max_depth=15,
                #         class_weight='balanced',
                #         random_state=42,
                #         n_jobs=-1
                #     ))
                # ])
                
                ### 4/24/25, EB: I am trying out a different model, using a Histogram-bsed Gradient Boosting Classifier.
                ### This is a bit of a shot in the dark, but I think it might be worth trying out. If it doesn't work, then oh well.
                # Compute sample weights once per fold
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('model', HistGradientBoostingClassifier(
                        loss='log_loss',           # For multi-class classification
                        learning_rate=0.05,#0.1
                        # class_weight='balanced',
                        max_iter=1000,#250
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        max_depth=7,
                        random_state=42
                    ))
                ])

                pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)
                
                print("y_test bins:", np.unique(y_test))
                print("y_pred bins:", np.unique(y_pred))
                
                # Binarize test labels for AUC
                y_test_bin = label_binarize(y_test, classes=class_labels)

                try:
                    auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
                except ValueError:
                    auc = np.nan  # If AUC can't be computed (e.g., only one class in test set)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'Accuracy': acc,
                    'F1_macro': f1,
                    'Recall_macro': recall,
                    'Precision_macro': precision,
                    'MacroAUC': auc
                })

                # === Logging for each fold ===
                log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
                log.write(classification_report(y_test, y_pred, digits=3))
                log.write("\nConfusion Matrix:\n")

                conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)
                conf_df = pd.DataFrame(
                    conf_mat,
                    index=[f'True {i}' for i in class_labels],
                    columns=[f'Pred {i}' for i in class_labels]
                )
                log.write(conf_df.to_string())
                log.write("\n\n")

            fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
            year_metrics = fold_df.mean().to_dict()
            year_metrics['Year'] = year
            metrics_all_years.append(year_metrics)

    metrics_df = pd.DataFrame(metrics_all_years)
    return metrics_df


#### Ordinal Modeling functions ###

def yearly_classification_prediction_ordinal(df, n_splits=5, log_path=log_path):
    """
    Classifies counties into log-normal-based RR bins (0–6) for each year using k-fold cross-validation.
    Logs detailed performance metrics and confusion matrices for every fold.
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
            y_class = df_year['rr_bin'].astype(int)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(drop='first'), ['county_class'])
                ], remainder='passthrough')

                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('model', LogisticIT(alpha=1.0))  # Regularization strength
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                mae = mean_absolute_error(y_test, y_pred)

                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'Accuracy': acc,
                    'F1_macro': f1,
                    'Recall_macro': recall,
                    'Precision_macro': precision,
                    'MAE': mae
                })

                # === Logging for each fold ===
                log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
                log.write(classification_report(y_test, y_pred, digits=3))
                log.write("\nConfusion Matrix:\n")

                labels_present = sorted(y_class.unique())
                conf_mat = confusion_matrix(y_test, y_pred, labels=labels_present)
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

def yearly_classification_prediction_ordinal_tuning(df, n_splits=5, log_path='ordinal_log.txt'):
    """
    Classifies counties into log-normal-based RR bins (0–6) for each year using k-fold cross-validation.
    Uses GridSearchCV to tune alpha parameter for LogisticAT in each fold.
    Logs detailed performance metrics and confusion matrices.
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
            y_class = df_year['rr_bin'].astype(int)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['county_class'])
                ], remainder='passthrough')

                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('model', LogisticAT())
                ])

                # === GridSearchCV ===
                param_grid = {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                }

                grid = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=3,  # inner CV for alpha tuning (can adjust)
                    scoring='accuracy',
                    n_jobs=1,
                    verbose=0
                )

                grid.fit(X_train, y_train)
                best_pipeline = grid.best_estimator_

                y_pred = best_pipeline.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                mae = mean_absolute_error(y_test, y_pred)

                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'Accuracy': acc,
                    'F1_weighted': f1,
                    'Recall_weighted': recall,
                    'Precision_weighted': precision,
                    'MAE': mae,
                    'Best_alpha': grid.best_params_['model__alpha']
                })

                # === Logging for each fold ===
                log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
                log.write(f"Best alpha: {grid.best_params_['model__alpha']}\n")
                log.write(classification_report(y_test, y_pred, digits=3))
                log.write("\nConfusion Matrix:\n")

                labels_present = sorted(y_class.unique())
                conf_mat = confusion_matrix(y_test, y_pred, labels=labels_present)
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

def ordinal_testing_cv(df, model_class, model_kwargs=None, n_splits=5):
    """
    Classifies counties into log-normal-based RR bins (0–6) for each year using k-fold cross-validation.
    Logs detailed performance metrics and confusion matrices for every fold.
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
            y_class = df_year['rr_bin'].astype(int)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(drop='first'), ['county_class'])
                ], remainder='passthrough')

                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('model', model_class(**(model_kwargs or {}))) #testing different ordinal models
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                mae = mean_absolute_error(y_test, y_pred)

                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'Accuracy': acc,
                    'F1_macro': f1,
                    'Recall_macro': recall,
                    'Precision_macro': precision,
                    'MAE': mae
                })

                # === Logging for each fold ===
                log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
                log.write(classification_report(y_test, y_pred, digits=3))
                log.write("\nConfusion Matrix:\n")

                labels_present = sorted(y_class.unique())
                conf_mat = confusion_matrix(y_test, y_pred, labels=labels_present)
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
    metrics_df['Model'] = model_class.__name__
    return metrics_df

def compare_mord_models(df, n_splits=5):
    models_to_try = [
        (LogisticAT, {'alpha': 1.0}),
        (LogisticIT, {'alpha': 1.0}),
        (OrdinalRidge, {'alpha': 1.0}),
    ]

    all_results = []

    for model_class, kwargs in models_to_try:
        print(f"\n🔍 Running {model_class.__name__}...")
        metrics_df = ordinal_testing_cv(df, model_class, model_kwargs=kwargs, n_splits=n_splits)
#        metrics_df = ordinal_testing_perbin_cv(df, model_class, model_kwargs=kwargs, n_splits=n_splits)
        all_results.append(metrics_df)

    return pd.concat(all_results, ignore_index=True)

from sklearn.metrics import precision_recall_fscore_support

def ordinal_testing_perbin_cv(df, model_class, model_kwargs=None, n_splits=5):
    """
    Classifies counties into log-normal-based RR bins (0–6) for each year using k-fold cross-validation.
    Logs detailed performance metrics and per-bin F1 scores for bins 0–3.
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
            y_class = df_year['rr_bin'].astype(int)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(drop='first'), ['county_class'])
                ], remainder='passthrough')

                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('model', model_class(**(model_kwargs or {})))
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # === Aggregate Metrics ===
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                mae = mean_absolute_error(y_test, y_pred)

                # === Per-bin F1 scores (for bins 0–3)
                bin_labels = [0, 1, 2, 3, 4, 5]  # or dynamically from y_class.unique()
                _, _, f1_per_bin, _ = precision_recall_fscore_support(
                    y_test, y_pred, labels=bin_labels, zero_division=0
                )

                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'Accuracy': acc,
                    'F1_macro': f1,
                    'Recall_macro': recall,
                    'Precision_macro': precision,
                    'MAE': mae,
                    'F1_bin0': f1_per_bin[0],
                    'F1_bin1': f1_per_bin[1],
                    'F1_bin2': f1_per_bin[2],
                    'F1_bin3': f1_per_bin[3]
                })

                # === Logging for each fold ===
                log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
                log.write(classification_report(y_test, y_pred, digits=3))
                log.write("\nConfusion Matrix:\n")

                labels_present = sorted(y_class.unique())
                conf_mat = confusion_matrix(y_test, y_pred, labels=labels_present)
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
    metrics_df['Model'] = model_class.__name__
    return metrics_df



### Plotting results functions:

def plot_results(metrics_df):
    """
    Plot accuracy, F1 (macro), and recall (macro) over prediction years.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=metrics_df, x='Year', y='Accuracy', label='Accuracy', marker='o')
    sns.lineplot(data=metrics_df, x='Year', y='F1_macro', label='F1 Score (Macro)', marker='s')
    sns.lineplot(data=metrics_df, x='Year', y='Recall_macro', label='Recall (Macro)', marker='^')
    sns.lineplot(data=metrics_df, x='Year', y='Precision_macro', label='Precision (Macro)', marker='x')
    sns.lineplot(data=metrics_df, x='Year', y='MAE', label='Mean Absolute Error', marker='D')

    plt.title("Model Classification Performance Over Time (Log-Normal Dist)")
    plt.ylabel("Score")
    plt.xlabel("Prediction Year")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_auc_scores_over_time(metrics_df):
    """
    Plots Macro-AUC scores over time using the summary metrics dataframe.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['Year'], metrics_df['MacroAUC'], marker='o', linewidth=2)
    plt.title("Macro-AUC of RR Bin Classifier Over Time")
    plt.xlabel("Year (for predicting year+1 RR bin)")
    plt.ylabel("Macro-AUC (One-vs-Rest)")
    plt.ylim(0.5, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ordinal_models(results_df):
    """
    Plots Accuracy and MAE over time for each ordinal regression model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Accuracy plot
    sns.lineplot(data=results_df, x='Year', y='Accuracy', hue='Model', marker='o', ax=axes[0])
    sns.lineplot(data=results_df, x='Year', y='F1_macro', hue='Model', marker='s', ax=axes[0])
    sns.lineplot(data=results_df, x='Year', y='Recall_macro', hue='Model', marker='^', ax=axes[0])
    axes[0].set_title("Accuracy Over Time")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Prediction Year")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # MAE plot
    sns.lineplot(data=results_df, x='Year', y='MAE', hue='Model', marker='s', ax=axes[1])
    axes[1].set_title("Mean Absolute Error Over Time")
    axes[1].set_ylabel("MAE (Bins)")
    axes[1].set_xlabel("Prediction Year")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_f1_per_bin_over_time(results_df):
    """
    Creates a 2×2 panel showing F1 scores over time for bins 0–3, split by model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

    bin_columns = ['F1_bin0', 'F1_bin1', 'F1_bin2', 'F1_bin3']
    bin_titles = ['Top 0.1% Risk (Bin 0)', 'Top 0.5% Risk (Bin 1)',
                  'Top 1% Risk (Bin 2)', 'Top 5% Risk (Bin 3)']

    for i, (col, title) in enumerate(zip(bin_columns, bin_titles)):
        ax = axes[i // 2, i % 2]
        sns.lineplot(data=results_df, x='Year', y=col, hue='Model', marker='o', ax=ax)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("Prediction Year")
        ax.set_ylabel("F1 Score")

    plt.tight_layout()
    plt.show()


def main():
    
    ### 4/25/25, EB: The following chunk of code is for the log-normal based classification model.
    ### By this I mean we sorted the bins using the log-normal distribution, and then used that to classify the counties.
    # print("Starting top 20% Random Forest classification and logging...")
    # all_years_df = prepare_yearly_prediction_data_adaptive(mortality_path)
    # print(all_years_df.groupby(['year', 'rr_bin']).size().unstack(fill_value=0))
    # metrics_df = yearly_classification_prediction_logging_AUC(all_years_df, n_splits=5, log_path=log_path)
    # plot_auc_scores_over_time(metrics_df)
    # print("✅ Classification and logging complete!")
    
    ### 4/25/25, EB: The following chunk of code is for the rank-order based classification model.
    ### The log-normal model was not working well because the bins were almost empty, so I am trying to use the rank-order model instead.
    ### I think that this might give us bins with more counties in them, and therefore enable actual classification.
    # print("Starting top 20% Random Forest classification and logging (rank-order)...")
    # all_years_df = prepare_yearly_prediction_data_rankorder(mortality_path)
    # #print(all_years_df.groupby(['year', 'rr_bin']).size().unstack(fill_value=0))
    # metrics_df = yearly_classification_prediction_logging_AUC(all_years_df, n_splits=4, log_path=log_path)
    # plot_auc_scores_over_time(metrics_df)
    # print("✅ Classification and logging complete!")

    ### 4/28/25, EB: The following chunk of code is for the ordinal regression model.
    ### I'm grasping at straws here, but I think since we're ultimately interested in which risk level the counties fall into, and these are ranked ordinal categroeis,
    ### it might be worth trying an ordinal regression model. I'm starting with the LogisticAT model, from mord, but I might try adapting LightGBM to do this like Chat suggested later.
    
    print("Starting top 20% Ordinal regression classification and logging...")
    all_years_df = prepare_yearly_prediction_data_adaptive(mortality_path)
#    print(all_years_df.groupby(['year', 'rr_bin']).size().unstack(fill_value=0))
    metrics_df = yearly_classification_prediction_ordinal(all_years_df, n_splits=5, log_path=log_path)
    plot_results(metrics_df)
    print("✅ Ordinal regression classification and logging complete!")

    ### 4/29/25, EB: Here I'm testing a few different ordinal regression models to see if they work better than the LogisticAT model.
    # print("Starting top 20% Ordinal Comparison...")
    # all_years_df = prepare_yearly_prediction_data_adaptive(mortality_path)
    # #all_years_df = prepare_yearly_prediction_data_lognormal(mortality_path)
    # #print(all_years_df.groupby(['year', 'rr_bin']).size().unstack(fill_value=0))
    # metrics_df = compare_mord_models(all_years_df)
    # plot_ordinal_models(metrics_df)
    # #plot_f1_per_bin_over_time(metrics_df)
    # print("✅ Ordinal regression classification and logging complete!")
    



if __name__ == '__main__':
    main()