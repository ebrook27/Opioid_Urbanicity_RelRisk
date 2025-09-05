### 8/19/25, EB: In the file EB_Logistic_Regression_Binary.py, I tested several different methods for logistic regression, to predict the relative risk level.
### Ordinary logistic regression is a binary model, but we're really interested in trying to predict risk at a much finer-grained level. I think ordinal logistic regression
### fits the bill, so I'm going to try that here. If this doesn't really work, then I think I'll pivot to trying XGBoost to do either ranking or ordinal regression.
### At the end of the day, this does feel like a ranking task? I know that there are ranking algorithms, but I'm not sure if our data would work as input well. One step at a time.

### Ran into the issue where a lot of counties have zero-mortality, so we need to be a little more clever in how we define our deciles. Updated add_ordinal_labels() function to pull out the zero MR counties into
### one category, and then use pd.qcut() to create deciles for the rest of the counties. This way we can still use the ordinal regression model, but it will be more robust to the zero-MR counties.

### 8/26/25, EB: Talked to Andrew, we need to not just use pd.qcut, we should use a lognormal to define the 10% bins of mortality. This is done in
### transform_to_categorical()

import numpy as np
np.int = int  # Monkey-patch for mord compatibility

from scipy.stats import lognorm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, mean_absolute_error
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import mord  # mord is a library for ordinal regression

#############
### Data preparation functions

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

def prepare_yearly_prediction_data():
    """Creates a long-format dataset for predicting next-year mortality using current-year SVI + county class."""
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2022))  # We predict mortality up to 2022

    # Load county category (static)
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)
    
    # Load mortality
    mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
    mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
    mort_df = mort_df.set_index('FIPS')
    
    # Load all SVI variables and reshape to long format per county-year
    svi_data = []
    for var in svi_variables:
        var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
        var_df = pd.read_csv(var_path, dtype={'FIPS': str})
        var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
        long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)

        # Extract year and filter relevant years
        long_df['year'] = long_df['year_var'].str.extract(r'(\d{4})').astype(int)
        long_df = long_df[long_df['year'].between(2010, 2021)]  # we predict 1 year ahead
        long_df = long_df.drop(columns='year_var')
        svi_data.append(long_df)

    # Merge all SVI variables on FIPS + year
    from functools import reduce
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Add mortality for year+1
    for y in years:
        mort_col = f'{y+1} MR'
        if mort_col not in mort_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'mortality_rate'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(mort_df[mort_col])

    # Add county class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows with any missing values
    svi_merged = svi_merged.dropna()

    return svi_merged


# def add_ordinal_labels(df, n_bins=10):
    # df = df.copy()
    # df['risk_category'] = pd.qcut(df['mortality_rate'], q=n_bins, labels=False)
    # return df

def add_ordinal_labels(df, n_bins=10):
    df = df.copy()
    
    # Split zero and non-zero mortality counties
    zero = df[df['mortality_rate'] == 0].copy()
    non_zero = df[df['mortality_rate'] > 0].copy()
    
    # Quantile bin the non-zero mortality rates
    non_zero['risk_category'] = pd.qcut(
        non_zero['mortality_rate'], 
        q=n_bins - 1,             # 1 bin already used for zero-mortality
        labels=False,
        duplicates='drop'         # avoid error from duplicate edges
    )
    
    # Shift categories to start at 1 instead of 0
    non_zero['risk_category'] += 1
    
    # Assign zero-risk category to zero-mortality counties
    zero['risk_category'] = 0
    
    # Combine and restore original order
    df_binned = pd.concat([zero, non_zero], axis=0).sort_index()
    df_binned['risk_category'] = df_binned['risk_category'].astype(int)
    
    return df_binned

def transform_to_categorical(df, svi_variables, bin_edges_dict=None):
    """
    Transforms the DataFrame into categorical format:
    - SVI variables binned into deciles (0–9)
    - Mortality rate binned into: 0 (no mortality) + 9 bins (log-normal)
    - County class cast as categorical

    Optionally accepts precomputed bin_edges_dict for mortality bins.
    Returns transformed DataFrame + bin_edges_dict for reuse.
    """
    df = df.copy()
    
    
    ### 8/26/25, EB: Keeping SVI as continuous for now
    # # ----------------------
    # # 1. Bin SVI variables into deciles
    # # ----------------------
    # for var in svi_variables:
    #     df[f'{var}_bin'] = pd.qcut(df[var], 10, labels=False, duplicates='drop')

    # ----------------------
    # 2. Bin mortality_rate
    # ----------------------
    if bin_edges_dict is None:
        bin_edges_dict = {}

    df['mortality_bin'] = np.nan

    for year in df['year'].unique():
        year_mask = (df['year'] == year)
        mort_year = df.loc[year_mask, 'mortality_rate']

        # Separate zeros
        is_zero = mort_year == 0
        is_positive = mort_year > 0

        df.loc[year_mask & is_zero, 'mortality_bin'] = 0  # Class 0: zero mortality

        # Fit log-normal to positive rates only
        if year not in bin_edges_dict:
            positive_rates = mort_year[is_positive]
            if len(positive_rates) < 20:
                print(f"⚠️ Year {year} has too few non-zero mortality entries for log-normal fit.")
                continue
            shape, loc, scale = lognorm.fit(positive_rates, floc=0)
            quantiles = np.linspace(0, 1, 10 + 1)  # 10 bins
            edges = lognorm.ppf(quantiles, shape, loc=loc, scale=scale)
            bin_edges_dict[year] = edges

        edges = bin_edges_dict[year]
        pos_values = mort_year[is_positive]
        binned = np.digitize(pos_values, edges[1:], right=False)  # Class 1–9
        df.loc[year_mask & is_positive, 'mortality_bin'] = binned

    ### 8/26/25, EB: Just int, not pandas Int64, to prevent issues in the models farther down 
    # df['mortality_bin'] = df['mortality_bin'].astype('Int64')
    df['mortality_bin'] = df['mortality_bin'].astype('int')
    
    # ---- Compute RR for each bin ----
    year_df = df.loc[year_mask].copy()
    total_deaths = year_df['mortality_rate'].sum()
    total_counties = year_df.shape[0]

    rr_map = {}
    for b in year_df['mortality_bin'].unique():
        bin_mask = year_df['mortality_bin'] == b
        deaths_in_bin = year_df.loc[bin_mask, 'mortality_rate'].sum()
        counties_in_bin = bin_mask.sum()

        if total_deaths > 0 and counties_in_bin > 0:
            rr = (deaths_in_bin / counties_in_bin) * (total_counties / total_deaths)
        else:
            rr = np.nan
        rr_map[b] = rr

    # Assign RR back to df for this year
    df.loc[year_mask, 'RR'] = df.loc[year_mask, 'mortality_bin'].map(rr_map)

    ### Keeping county_class as numerical for now
    # # ----------------------
    # # 3. Treat county class as categorical
    # # ----------------------
    # df['county_class'] = df['county_class'].astype('category')

    return df, bin_edges_dict


#############
### Model functions

def run_ordinal_logistic_regression(data_df, n_splits=5):
    '''
    Runs ordinal logistic regression with k-fold CV, predicting risk category using SVI + county class. Doesn't account for temporal aspect, so uses all data.
    '''

    features = [col for col in DATA if col != 'Mortality'] + ['county_class']
    target = 'risk_category'

    X = data_df[features].copy()
    y = data_df[target].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), ['county_class'])
        ])

        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', mord.LogisticAT(alpha=1.0))  # you can tune alpha
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    print("🔍 Classification Report (Ordinal):\n", classification_report(all_true, all_pred))
    return all_true, all_pred

def run_ordinal_logistic_sliding_window(data_df, n_splits=5):
    '''
    Runs ordinal logistic regression with a sliding window approach, predicting next-year risk category using current-year features.
    '''
    
    
    features = [col for col in DATA if col != 'Mortality'] + ['county_class']
    target = 'risk_category'
    
    years = sorted(data_df['year'].unique())
    all_true = []
    all_pred = []
    all_index = []

    for year in years:
        train_df = data_df[data_df['year'] == year].copy()
        test_df = data_df[data_df['year'] == year].copy()
        #print(f"train_df columns: ", train_df.columns)
        #print(f"test_df.columns: ", test_df.columns)
        
        if len(test_df) == 0:
            continue  # Skip if next year's data is missing

        X = train_df[features].copy()
        y = train_df[target].copy()
        test_X = test_df[features].copy()
        test_y = test_df[target].copy()
        
        #print(f"X.columns: ", X.columns)
        #print(f"y.columns: ", y.columns)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_true = []
        fold_pred = []
        fold_index = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['county_class'])
            ], remainder='passthrough')

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('clf', mord.LogisticAT(alpha=1.0))
            ])

            pipeline.fit(X_train, y_train)

            # Predict for the corresponding subset of the next year's data
            # Match by index to avoid length mismatch
            y_pred = pipeline.predict(test_X)

            fold_true.extend(test_y)
            fold_pred.extend(y_pred)
            fold_index.extend(test_X.index)

        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        all_index.extend(fold_index)

        print(f"✅ Year {year} → {year+1} done")

    # Final evaluation
    print("🔍 Classification Report (Ordinal Sliding Window):\n",
          classification_report(all_true, all_pred))

    # Optional: build a results DataFrame
    results_df = data_df.loc[all_index].copy()
    results_df['True'] = all_true
    results_df['Predicted'] = all_pred

    return results_df

from sklearn.metrics import classification_report, mean_absolute_error
from scipy.stats import spearmanr, kendalltau
import numpy as np

def run_ordinal_logistic_sliding_window_full_metrics(data_df, n_splits=5, top_k=0.1):
    """
    Runs ordinal logistic regression (LogisticAT from mord) with a sliding window approach,
    predicting next-year risk category using current-year features.

    Evaluates with classification metrics, MAE, rank correlations, and Top-K overlap.
    """

    features = [col for col in DATA if col != 'Mortality'] + ['county_class']
    # target = 'risk_category'
    target = 'mortality_bin'
    
    years = sorted(data_df['year'].unique())
    all_true = []
    all_pred = []
    all_index = []

    for year in years:
        train_df = data_df[data_df['year'] == year].copy()
        test_df = data_df[data_df['year'] == year].copy()

        if len(test_df) == 0:
            continue  # Skip if next year's data is missing

        X = train_df[features].copy()
        y = train_df[target].copy()
        test_X = test_df[features].copy()
        test_y = test_df[target].copy()

        ### 9/4/25, EB: Correcting KFold splitting to account for class imbalances.
        # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_true = []
        fold_pred = []
        fold_index = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['county_class'])
            ], remainder='passthrough')

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('clf', mord.LogisticAT(alpha=0.5))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(test_X)

            fold_true.extend(test_y)
            fold_pred.extend(y_pred)
            fold_index.extend(test_X.index)

        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        all_index.extend(fold_index)
        print(f"✅ Year {year} → {year+1} done")

    # ---- Evaluation ----
    print("\n🔍 Classification Report (Ordinal Sliding Window):\n",
          classification_report(all_true, all_pred))

    # MAE on ordinal bins
    mae = mean_absolute_error(all_true, all_pred)
    print(f"\n📉 Mean Absolute Error (bins): {mae:.4f}")

    # Ranking metrics
    spearman_corr, _ = spearmanr(all_true, all_pred)
    kendall_corr, _ = kendalltau(all_true, all_pred)
    print(f"📈 Spearman rank correlation: {spearman_corr:.4f}")
    print(f"📈 Kendall's tau: {kendall_corr:.4f}")

    # Top-K hit rate
    k = int(len(all_true) * top_k)
    true_top_idx = np.argsort(all_true)[-k:]
    pred_top_idx = np.argsort(all_pred)[-k:]
    overlap = len(set(true_top_idx).intersection(set(pred_top_idx)))
    topk_hit_rate = overlap / k if k > 0 else np.nan
    print(f"🎯 Top-{int(top_k*100)}% hit rate: {topk_hit_rate:.4f}")

    # ---- Results DF ----
    results_df = data_df.loc[all_index].copy()
    results_df['True'] = all_true
    results_df['Predicted'] = all_pred

    return results_df

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, kendalltau
import mord
import numpy as np
import pandas as pd

def run_ordinal_logistic_sliding_window_full_metrics_per_year(data_df, n_splits=5, top_k=0.1):
    """
    Runs ordinal logistic regression (LogisticAT from mord) with a sliding window approach,
    predicting next-year risk category using current-year features.

    Evaluates with classification metrics, MAE, rank correlations, and Top-K overlap.
    Prints per-year classification reports (averaged across folds)
    and an overall report across all years.
    """

    features = [col for col in DATA if col != 'Mortality'] + ['county_class']
    target = 'mortality_bin'
    
    years = sorted(data_df['year'].unique())
    all_true = []
    all_pred = []
    all_index = []

    per_year_results = {}

    for year in years:
        train_df = data_df[data_df['year'] == year].copy()
        test_df = data_df[data_df['year'] == year].copy()

        if len(test_df) == 0:
            continue  # Skip if next year's data is missing

        X = train_df[features].copy()
        y = train_df[target].copy()
        test_X = test_df[features].copy()
        test_y = test_df[target].copy()

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_true = []
        fold_pred = []
        fold_index = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['county_class'])
            ], remainder='passthrough')

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('clf', mord.LogisticAT(alpha=0.5))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(test_X)  # predictions on full test year

            fold_true.extend(test_y)
            fold_pred.extend(y_pred)
            fold_index.extend(test_X.index)

        # store overall per-year results
        per_year_results[year] = {
            "y_true": np.array(fold_true),
            "y_pred": np.array(fold_pred),
            "indices": fold_index
        }

        # extend global trackers
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        all_index.extend(fold_index)

        print(f"✅ Year {year} → {year+1} done")

    # ---- Per-year classification reports ----
    print("\n📊 Per-Year Classification Reports (averaged across folds):")
    for year, results in per_year_results.items():
        print(f"\nYear {year} → {year+1}")
        print(classification_report(results["y_true"], results["y_pred"], zero_division=0))

    # ---- Overall evaluation ----
    print("\n🔍 Overall Classification Report (all years combined):\n",
          classification_report(all_true, all_pred, zero_division=0))

    # MAE on ordinal bins
    mae = mean_absolute_error(all_true, all_pred)
    print(f"\n📉 Overall Mean Absolute Error (bins): {mae:.4f}")

    # Ranking metrics
    spearman_corr, _ = spearmanr(all_true, all_pred)
    kendall_corr, _ = kendalltau(all_true, all_pred)
    print(f"📈 Spearman rank correlation: {spearman_corr:.4f}")
    print(f"📈 Kendall's tau: {kendall_corr:.4f}")

    # Top-K hit rate
    k = int(len(all_true) * top_k)
    true_top_idx = np.argsort(all_true)[-k:]
    pred_top_idx = np.argsort(all_pred)[-k:]
    overlap = len(set(true_top_idx).intersection(set(pred_top_idx)))
    topk_hit_rate = overlap / k if k > 0 else np.nan
    print(f"🎯 Top-{int(top_k*100)}% hit rate: {topk_hit_rate:.4f}")

    # ---- Results DF ----
    results_df = data_df.loc[all_index].copy()
    results_df['True'] = all_true
    results_df['Predicted'] = all_pred

    return results_df



def run_binary_logistic_sliding_window_full_metrics(data_df, n_splits=5, top_k=0.1):
    """
    Runs ordinal logistic regression (LogisticAT from mord) with a sliding window approach,
    but collapses mortality_bin into two classes:
      - Class 1: mortality_bin == 9
      - Class 0: all other bins
    Evaluates with classification metrics and related metrics.
    """

    features = [col for col in DATA if col != 'Mortality'] + ['county_class']
    target = 'mortality_bin'
    
    years = sorted(data_df['year'].unique())
    all_true, all_pred, all_index = [], [], []

    for year in years:
        train_df = data_df[data_df['year'] == year].copy()
        test_df = data_df[data_df['year'] == year].copy()

        if len(test_df) == 0:
            continue  # Skip if next year's data is missing

        X = train_df[features].copy()
        # collapse target: 1 if mortality_bin == 9, else 0
        y = (train_df[target] == 9).astype(int)
        print("Unique classes in y:", y.unique())
        print("Counts:\n", y.value_counts())
        test_X = test_df[features].copy()
        test_y = (test_df[target] == 9).astype(int)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_true, fold_pred, fold_index = [], [], []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['county_class'])
            ], remainder='passthrough')

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('clf', mord.LogisticAT(alpha=0.5))  # effectively logistic regression now
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(test_X)

            fold_true.extend(test_y)
            fold_pred.extend(y_pred)
            fold_index.extend(test_X.index)

        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        all_index.extend(fold_index)

        print(f"✅ Year {year} → {year+1} done")

    # ---- Evaluation ----
    print("\n🔍 Classification Report (Binary Sliding Window):\n",
          classification_report(all_true, all_pred))

    # Accuracy / MAE (for completeness, MAE is just misclassification rate here)
    mae = mean_absolute_error(all_true, all_pred)
    print(f"\n📉 Mean Absolute Error (0/1): {mae:.4f}")

    # Ranking metrics (less meaningful for binary, but still ok)
    spearman_corr, _ = spearmanr(all_true, all_pred)
    kendall_corr, _ = kendalltau(all_true, all_pred)
    print(f"📈 Spearman rank correlation: {spearman_corr:.4f}")
    print(f"📈 Kendall's tau: {kendall_corr:.4f}")

    # Top-K hit rate (here, effectively how many "1"s you catch if you pick top-k by prediction)
    k = int(len(all_true) * top_k)
    true_top_idx = np.argsort(all_true)[-k:]
    pred_top_idx = np.argsort(all_pred)[-k:]
    overlap = len(set(true_top_idx).intersection(set(pred_top_idx)))
    topk_hit_rate = overlap / k if k > 0 else np.nan
    print(f"🎯 Top-{int(top_k*100)}% hit rate: {topk_hit_rate:.4f}")

    # ---- Results DF ----
    results_df = data_df.loc[all_index].copy()
    results_df['True'] = all_true
    results_df['Predicted'] = all_pred

    return results_df

def run_ordinal_logistic_expanding_window(data_df, start_year=2010, n_splits=5):
    features = [col for col in DATA if col != 'Mortality'] + ['county_class']
    target = 'risk_category'

    years = sorted(data_df['year'].unique())
    all_true = []
    all_pred = []
    all_index = []

    for i, year in enumerate(years):
        if year <= start_year:
            continue

        # Use all years up to year-1 for training
        train_df = data_df[data_df['year'] < year].copy()
        test_df = data_df[data_df['year'] == year].copy()

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[features].copy()
        y_train = train_df[target].copy()
        X_test = test_df[features].copy()
        y_test = test_df[target].copy()

        # Use KFold on train set for robustness
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_preds = []
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, _ = y_train.iloc[train_idx], y_train.iloc[val_idx]

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['county_class'])
            ], remainder='passthrough')

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('clf', mord.LogisticAT(alpha=1.0))
            ])

            pipeline.fit(X_fold_train, y_fold_train)

        # Fit final model on full training set
        final_pipeline = Pipeline([
            ('prep', ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['county_class'])
            ], remainder='passthrough')),
            ('clf', mord.LogisticAT(alpha=1.0))
        ])
        final_pipeline.fit(X_train, y_train)

        y_pred = final_pipeline.predict(X_test)

        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_index.extend(X_test.index)

        print(f"📈 Trained on years <= {year-1} → predicted year {year}")

    # Final evaluation
    print("\n🔍 Classification Report (Ordinal Expanding Window):\n",
          classification_report(all_true, all_pred))

    results_df = data_df.loc[all_index].copy()
    results_df['True'] = all_true
    results_df['Predicted'] = all_pred

    return results_df


#############
#### Testing different mord models

def run_all_mord_models(data_df, n_splits=5):
    """
    Run multiple mord ordinal regression models with a sliding window approach.
    Predict next-year risk category using current-year features.
    """

    # Define models to evaluate
    mord_models = {
        "LogisticAT": mord.LogisticAT(alpha=1.0),
        "LogisticIT": mord.LogisticIT(alpha=1.0),
        "LogisticSE": mord.LogisticSE(alpha=1.0),
        #"LogisticGrid": mord.LogisticGrid(alpha=1.0),
        "OrdinalRidge": mord.OrdinalRidge(alpha=1.0)
    }

    features = [col for col in data_df.columns if col not in ['mortality_rate', 'risk_category', 'year', 'FIPS']]# + ['county_class']
    target = 'risk_category'
    years = sorted(data_df['year'].unique())

    model_results = {}

    for model_name, model in mord_models.items():
        print(f"\n🚀 Running model: {model_name}")
        all_true, all_pred, all_index = [], [], []

        for year in years:
            train_df = data_df[data_df['year'] == year].copy()
            test_df = data_df[data_df['year'] == year].copy()

            # print(f"train_df columns: ", train_df.columns)
            # print(f"test_df.columns: ", test_df.columns)

            if len(test_df) == 0:
                continue

            X = train_df[features].copy()
            y = train_df[target].copy()
            test_X = test_df[features].copy()
            test_y = test_df[target].copy()
            
            # print(f"X.columns: ", X.columns)
            # print(f"y.columns: ", y.columns)

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(drop='first'), ['county_class'])
                ], remainder='passthrough')

                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('clf', model)
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(test_X)

                all_true.extend(test_y)
                all_pred.extend(y_pred)
                all_index.extend(test_X.index)

            print(f"✅ Year {year} → {year+1} done")

        # Collect results
        report = classification_report(all_true, all_pred, output_dict=True)
        results_df = data_df.loc[all_index].copy()
        results_df['True'] = all_true
        results_df['Predicted'] = all_pred

        model_results[model_name] = {
            "report": report,
            "results_df": results_df
        }

        print(f"📊 {model_name} report:\n", classification_report(all_true, all_pred))

    return model_results

def summarize_mord_results(model_results):
    """
    Summarize performance metrics (Accuracy, Macro-F1, Weighted-F1, MAE)
    across all mord models.
    
    Parameters
    ----------
    model_results : dict
        Output from run_all_mord_models(), containing reports and results_df 
        for each model.
    
    Returns
    -------
    summary_df : pd.DataFrame
        Table with models as rows and metrics as columns.
    """

    rows = []
    for model_name, res in model_results.items():
        y_true = res["results_df"]["True"]
        y_pred = res["results_df"]["Predicted"]

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        mae = mean_absolute_error(y_true, y_pred)

        rows.append({
            "Model": model_name,
            "Accuracy": acc,
            "MacroF1": macro_f1,
            "WeightedF1": weighted_f1,
            "MAE": mae
        })

    summary_df = pd.DataFrame(rows).set_index("Model")
    print("\n📊 Model Comparison Summary:\n")
    print(summary_df.round(4))

    return summary_df


#### Running a gridsearch over alpha values for all the mord models:

def run_all_mord_models_with_alpha_search(data_df, n_splits=5, alpha_grid=None):
    """
    Run multiple mord ordinal regression models with a sliding window approach,
    tuning alpha parameter over a grid.
    """

    if alpha_grid is None:
        alpha_grid = [0.01, 0.1, 1, 10, 100]

    # Define models to evaluate (only models with alpha parameter)
    mord_models = {
        "LogisticAT": mord.LogisticAT,
        "LogisticIT": mord.LogisticIT,
        "LogisticSE": mord.LogisticSE,
        "OrdinalRidge": mord.OrdinalRidge
    }

    features = [col for col in data_df.columns 
                if col not in ['mortality_rate', 'mortality_bin', 'year', 'FIPS']]
    target = 'mortality_bin'
    years = sorted(data_df['year'].unique())

    model_results = {}

    for model_name, model_class in mord_models.items():
        print(f"\n🚀 Running model: {model_name}")

        best_alpha = None
        best_macro_f1 = -np.inf
        best_results = None

        for alpha in alpha_grid:
            print(f"   🔎 Trying alpha={alpha}")
            all_true, all_pred, all_index = [], [], []

            for year in years[:6]:
                train_df = data_df[data_df['year'] == year].copy()
                test_df = data_df[data_df['year'] == year].copy()

                if len(test_df) == 0:
                    continue

                X = train_df[features].copy()
                y = train_df[target].copy()
                test_X = test_df[features].copy()
                test_y = test_df[target].copy()

                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    preprocessor = ColumnTransformer([
                        ('cat', OneHotEncoder(drop='first'), ['county_class'])
                    ], remainder='passthrough')

                    pipeline = Pipeline([
                        ('prep', preprocessor),
                        ('clf', model_class(alpha=alpha))
                    ])

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(test_X)

                    all_true.extend(test_y)
                    all_pred.extend(y_pred)
                    all_index.extend(test_X.index)

            macro_f1 = f1_score(all_true, all_pred, average="macro")

            # if macro_f1 > best_macro_f1:
            #     best_macro_f1 = macro_f1
            #     best_alpha = alpha
            #     best_results = {
            #         "report": classification_report(all_true, all_pred, output_dict=True),
            #         "results_df": data_df.loc[all_index].assign(True=all_true, Predicted=all_pred)
            #     }
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_alpha = alpha

                tmp_df = data_df.loc[all_index].copy()
                tmp_df["True"] = all_true
                tmp_df["Predicted"] = all_pred

                best_results = {
                    "report": classification_report(all_true, all_pred, output_dict=True),
                    "results_df": tmp_df
                }


            print(f"      Macro-F1={macro_f1:.4f}")

        # Store best results for this model
        model_results[model_name] = {
            "best_alpha": best_alpha,
            "best_macro_f1": best_macro_f1,
            "report": best_results["report"],
            "results_df": best_results["results_df"]
        }

        print(f"✅ {model_name}: best alpha={best_alpha}, Macro-F1={best_macro_f1:.4f}")

    return model_results


#########################################
def main():
    print('Running Ordinal Logistic Regression model:')
    # Prepare data
    df = prepare_yearly_prediction_data()
    svi_variables = [v for v in DATA if v != 'Mortality']
    
    # Label high-risk counties
    #df = add_ordinal_labels(df, n_bins=10) # 10 bins -> deciles  
    df, bin_edges_dict = transform_to_categorical(df, svi_variables)
    # print(df.tail())
    # print(f"Unique Risk Scores: {df['RR'].unique()}")
    

    # Run logistic regression
    results_df = run_ordinal_logistic_sliding_window_full_metrics_per_year(df)
    #results_df = run_ordinal_logistic_sliding_window_full_metrics(df)
    
    # results_df['correct'] = results_df['True'] == results_df['Predicted']
    # accuracy_by_year = results_df.groupby('year')['correct'].mean()
    # accuracy_by_year.plot(kind='bar')
    #print('Model prediction complete.')

# def main():
#     df = prepare_yearly_prediction_data()
#     df = add_ordinal_labels(df, n_bins=10)
# #    print(df.columns)
#     # Run models
#     results = run_all_mord_models(df)

#     # Summarize metrics
#     summary = summarize_mord_results(results)

### Tuning loop
# def main():
#     df = prepare_yearly_prediction_data()
#     svi_variables = [v for v in DATA if v != 'Mortality']
    
#     df, bin_edges_dict = transform_to_categorical(df, svi_variables)
    
#     results = run_all_mord_models_with_alpha_search(df, alpha_grid=[0.01, 0.1, 1, 10, 100, 1000])

if __name__ == "__main__":
    main()