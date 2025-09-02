### 4/8/25, EB:  This script is based off the rel_risk_regression_model.py file, but after speaking with Andrew,
### I realized I had sort of misunderstood the goal. Rather than predicting mortality rates based off the SVI data (plus county category),
### and then converting the predictions into relative risk scores, we should instead convert the mortality data into relative risk scores directly.
### Then we can stratify these scores into 20 bins, and use those as our target variable.
### This turns it into a classification problem rather than a regression problem, which will hopefully make things easier,
### and more meaningful in terms of interpreting the results.

### So, the pipeline here is as follows: first we will convert mortality rates into relative risk scores. Then we will
### stratify these scores into 20 risk levels (or bins). Then we will use the SVI data (plus county category) to predict these risk levels.
### Finally, we will evaluate the model using the same metrics as before (RMSE, MAE, R2, etc.), but also using
### classification metrics like accuracy, precision, recall, etc.

### 4/15/25, EB: I was able to fit a distribution to the mortality data for each year, the log-normal distribution fit best (This matches what Andrew found).
### Then I used the distribution to bin the mortality data into 20 bins (based on every 5th percentile from the distribution), and then saved the results.
### Then I imported those bins here instead of the naive bins I used before, and classified the counties using SVI data + county category into those bins.
### The results are really good, all the scores (accuracy, precision, recall, and F1) are above 0.9 for all years.
### What I'm going to try now is to extract feature importance across the folds, which is easy to do using the RF classifier.
### I will then plot the feature importance for each year, and see if there are any trends over time. 

### 4/15/25, EB: The results are terrible actually, I had made a mistake with the input features. Now that I have it corrected, the scores were all around 0.1-0.2
### I think that unless we look at fewer risk levels, or rethink the way we bin the levels, this isn't the move.
### In the file EB_rel_risk_model_redux.py, I switch to a regression model to predict the relative risk scores, and that seems to work somewhat better.

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings("ignore", message=".*use_inf_as_na option is deprecated.*")


# custom_levels_path = 'Data\Mortality\Final Files\Mortality_relative_risk_custom_levels.csv'
# even_levels_path = 'Data\Mortality\Final Files\Mortality_relative_risk_levels.csv'

lognormal_levels_path = 'Data\Mortality\Final Files\Mortality_lognormal_binned_RR.csv'

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

# def prepare_yearly_prediction_data(mortality_path):
#     """Creates a long-format dataset for predicting next-year mortality using current-year SVI + county class."""
#     svi_variables = [v for v in DATA if v != 'Mortality']
#     years = list(range(2010, 2022))  # We predict mortality up to 2022

#     # Load county category (static)
#     nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
#     nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
#     nchs_df = nchs_df.set_index('FIPS')
#     nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)
    
#     # Load mortality
#     rr_df = pd.read_csv(mortality_path, dtype={'FIPS': str})
#     rr_df['FIPS'] = rr_df['FIPS'].str.zfill(5)
#     rr_df = rr_df.set_index('FIPS')
    
#     # Load all SVI variables and reshape to long format per county-year
#     svi_data = []
#     for var in svi_variables:
#         var_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
#         var_df = pd.read_csv(var_path, dtype={'FIPS': str})
#         var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
#         long_df = var_df.melt(id_vars='FIPS', var_name='year_var', value_name=var)

#         # Extract year and filter relevant years
#         long_df['year'] = long_df['year_var'].str.extract(r'(\d{4})').astype(int)
#         long_df = long_df[long_df['year'].between(2010, 2021)]  # we predict 1 year ahead
#         long_df = long_df.drop(columns='year_var')
#         svi_data.append(long_df)

#     # Merge all SVI variables on FIPS + year
#     from functools import reduce
#     svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

#     # 🔁 Add binned relative risk level for year+1
#     for y in years:
#         rr_col = f'{y+1} RR_Level'
#         if rr_col not in rr_df.columns:
#             continue
#         svi_merged.loc[svi_merged['year'] == y, 'rr_bin'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(rr_df[rr_col])
    
#     svi_merged['rr_bin'] = svi_merged['rr_bin'].astype(int)

#     # Add county class
#     svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

#     # Drop rows with any missing values
#     svi_merged = svi_merged.dropna()

#     return svi_merged

def prepare_yearly_prediction_data_lognormal(mortality_path):
    """Creates a long-format dataset for predicting next-year RR level using current-year SVI + county class."""
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2022))  # We predict for mortality years 2011–2022

    # Load county category
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

    # Load mortality RR level data
    rr_df = pd.read_csv(mortality_path, dtype={'FIPS': str})
    rr_df['FIPS'] = rr_df['FIPS'].str.zfill(5)
    rr_df = rr_df.set_index('FIPS')

    # Load and reshape all SVI variables
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

    # Merge SVI variables on FIPS + year
    from functools import reduce
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Add RR level for year+1
    for y in years:
        rr_col = f'{y+1}_RR_Level'
        if rr_col not in rr_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'rr_bin'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(rr_df[rr_col])

    # Convert rr_bin to integer (from float or object)
    svi_merged['rr_bin'] = svi_merged['rr_bin'].astype(int)
    
    # ## Troubleshooting!
    # # Safely assign rr_bin from each year's RR_Level column
    # rr_bin_assigned = False
    # for y in years:
    #     rr_col = f'{y+1}_RR_Level'
    #     if rr_col not in rr_df.columns:
    #         print(f"⚠️ Missing column: {rr_col} in RR file.")
    #         continue
    #     mask = svi_merged['year'] == y
    #     if mask.sum() == 0:
    #         print(f"⚠️ No SVI data for year {y}, skipping.")
    #         continue
    #     mapped_bins = svi_merged.loc[mask, 'FIPS'].map(rr_df[rr_col])
    #     if mapped_bins.notna().sum() == 0:
    #         print(f"⚠️ No matching FIPS for year {y} in RR column {rr_col}.")
    #         continue
    #     svi_merged.loc[mask, 'rr_bin'] = mapped_bins
    #     rr_bin_assigned = True

    # if not rr_bin_assigned:
    #     raise ValueError("❌ No rr_bin values assigned — check your RR file and column names.")
    # else:
    #     svi_merged['rr_bin'] = svi_merged['rr_bin'].astype(int)


    # Add county class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows with missing values
    svi_merged = svi_merged.dropna()

    return svi_merged


# def yearly_classification_prediction(df, n_splits=5):
#     """
#     Classifies counties into binned relative risk classes (0–n_bins) for each year.
#     Uses cross-validation within each year for robustness.
#     """
#     metrics_all_years = []

#     for year in range(2010, 2021):
#         print(f"🔁 Processing year {year} → predicting RR bin for year {year+1}")
#         df_year = df[df['year'] == year].copy()

#         if df_year.empty:
#             print(f"⚠️ Skipping year {year}: no data.")
#             continue

#         feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'relative_risk_score']]
#         X = df_year[feature_cols]
#         y_class = df_year['rr_bin']  # Already an integer class label

#         skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#         fold_metrics = []

#         for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
#             X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#             y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]
            
#             # Preprocessing + classifier pipeline
#             preprocessor = ColumnTransformer([
#                 ('cat', OneHotEncoder(drop='first'), ['county_class'])
#             ], remainder='passthrough')

#             pipeline = Pipeline([
#                 ('prep', preprocessor),
#                 ('model', RandomForestClassifier(
#                     n_estimators=250,
#                     max_depth=15,
#                     class_weight='balanced',
#                     random_state=42,
#                     n_jobs=-1
#                 ))
#             ])

#             pipeline.fit(X_train, y_train)
#             y_pred = pipeline.predict(X_test)
            
#             acc = accuracy_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
#             recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
#             precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

#             fold_metrics.append({'fold': fold_idx+1, 'Accuracy': acc, 'F1_macro': f1, 'Recall_macro': recall, 'Precision_macro': precision})

#         # Aggregate metrics for the year
#         fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
#         year_metrics = fold_df.mean().to_dict()
#         year_metrics['Year'] = year
#         metrics_all_years.append(year_metrics)

#     metrics_df = pd.DataFrame(metrics_all_years)
#     return metrics_df

def yearly_classification_prediction_logging(df, n_splits=5, n_bins=20, log_path='RR_lognormal_classification_log.txt'):
    """
    Classifies counties into binned relative risk classes (0–n_bins) for each year.
    Uses cross-validation within each year for robustness.
    Logs classification reports and confusion matrices.
    """
    metrics_all_years = []

    with open(log_path, 'w') as log:  # Open log file once
        for year in range(2010, 2021):
            print(f"🔁 Processing year {year} → predicting RR bin for year {year+1}")
            df_year = df[df['year'] == year].copy()

            if df_year.empty:
                print(f"⚠️ Skipping year {year}: no data.")
                continue

            feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'relative_risk_score', 'rr_bin']]
            X = df_year[feature_cols]
            y_class = df_year['rr_bin']

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

                # Preprocessing + classifier pipeline
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
                conf_mat = confusion_matrix(y_test, y_pred, labels=np.arange(n_bins))
                conf_df = pd.DataFrame(conf_mat, 
                    index=[f'True {i}' for i in range(n_bins)],
                    columns=[f'Pred {i}' for i in range(n_bins)])
                log.write(conf_df.to_string())
                log.write("\n\n")

        # Average fold scores for the year
            fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
            year_metrics = fold_df.mean().to_dict()
            year_metrics['Year'] = year
            metrics_all_years.append(year_metrics)

    metrics_df = pd.DataFrame(metrics_all_years)
    
    return metrics_df

def yearly_classification_prediction_targeted_logging(df, n_splits=5, n_bins=20, log_path='RR_lognormal_classification_log.txt'):
    """
    Classifies counties into binned relative risk classes (0–n_bins) for each year.
    Uses cross-validation within each year for robustness.
    Logs classification reports, confusion matrices, and target-class performance.
    """
    metrics_all_years = []
    target_class = 19  # Change this if you want to monitor a different class

    with open(log_path, 'w') as log:
        for year in range(2010, 2021):
            print(f"🔁 Processing year {year} → predicting RR bin for year {year+1}")
            df_year = df[df['year'] == year].copy()

            if df_year.empty:
                print(f"⚠️ Skipping year {year}: no data.")
                continue

            feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'relative_risk_score', 'rr_bin']]
            X = df_year[feature_cols]
            y_class = df_year['rr_bin']

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

                # Fold-wide metrics
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

                # === Logging ===
                log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
                log.write(classification_report(y_test, y_pred, digits=3))
                log.write("\nConfusion Matrix:\n")
                conf_mat = confusion_matrix(y_test, y_pred, labels=np.arange(n_bins))
                conf_df = pd.DataFrame(conf_mat, 
                    index=[f'True {i}' for i in range(n_bins)],
                    columns=[f'Pred {i}' for i in range(n_bins)])
                log.write(conf_df.to_string())
                log.write("\n")

                # === Log target class (e.g. top risk bin) performance ===
                prec_arr, rec_arr, f1_arr, support_arr = precision_recall_fscore_support(
                    y_test, y_pred, labels=[target_class], average=None, zero_division=0
                )

                log.write(f"\n🎯 Class {target_class} Metrics — Fold {fold_idx+1}:\n")
                log.write(f"Precision: {prec_arr[0]:.3f}, Recall: {rec_arr[0]:.3f}, F1: {f1_arr[0]:.3f}, Support: {support_arr[0]}\n\n")

            # Average across folds
            fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
            year_metrics = fold_df.mean().to_dict()
            year_metrics['Year'] = year
            metrics_all_years.append(year_metrics)

    return pd.DataFrame(metrics_all_years)

def yearly_classification_prediction_featimp(df, n_splits=5):
    metrics_all_years = []
    all_feature_importances = []

    for year in range(2010, 2021):
        print(f"🔁 Processing year {year} → predicting RR bin for year {year+1}")
        df_year = df[df['year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            continue

        feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'relative_risk_score', 'rr_bin']]
        X = df_year[feature_cols]
        #print(X.columns.tolist())
        y_class = df_year['rr_bin']  # Already an integer class label

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        fold_importances = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

            # Preprocessing + classifier pipeline
            categorical = ['county_class']
            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), categorical)
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

            # Store metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            fold_metrics.append({'fold': fold_idx+1, 'Accuracy': acc, 'F1_macro': f1, 'Recall_macro': recall, 'Precision_macro': precision})

            # Extract feature importances
            rf_model = pipeline.named_steps['model']
            ohe = pipeline.named_steps['prep'].named_transformers_['cat']
            ohe_feature_names = ohe.get_feature_names_out(categorical)
            remainder_cols = [col for col in X.columns if col not in categorical]
            full_feature_names = list(ohe_feature_names) + remainder_cols
            importance_df = pd.DataFrame({
                'Feature': full_feature_names,
                'Importance': rf_model.feature_importances_,
                'Year': year,
                'Fold': fold_idx + 1
            })
            fold_importances.append(importance_df)

        # Aggregate metrics
        fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
        year_metrics = fold_df.mean().to_dict()
        year_metrics['Year'] = year
        metrics_all_years.append(year_metrics)

        # Combine importances for the year
        all_feature_importances.append(pd.concat(fold_importances))

    metrics_df = pd.DataFrame(metrics_all_years)
    importance_df = pd.concat(all_feature_importances)

    return metrics_df, importance_df

def yearly_classification_prediction_logging_undersamp(df, n_splits=5, n_bins=20, log_path='RR_lognormal_classification_log.txt'):
    """
    Classifies counties into binned relative risk classes (0–n_bins) for each year.
    Uses cross-validation within each year for robustness.
    Logs classification reports and confusion matrices.
    Applies random undersampling to address class imbalance.
    """
    metrics_all_years = []

    with open(log_path, 'w') as log:  # Open log file once
        for year in range(2010, 2021):
            print(f"🔁 Processing year {year} → predicting RR bin for year {year+1}")
            df_year = df[df['year'] == year].copy()

            if df_year.empty:
                print(f"⚠️ Skipping year {year}: no data.")
                continue

            feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'relative_risk_score', 'rr_bin']]
            X = df_year[feature_cols]
            y_class = df_year['rr_bin']

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

                # === Undersampling step ===
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_train, y_train)

                # Preprocessing + classifier pipeline
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

                # Logging
                log.write(f"=== Year {year}, Fold {fold_idx+1} ===\n")
                log.write(classification_report(y_test, y_pred, digits=3))
                log.write("\nConfusion Matrix:\n")
                conf_mat = confusion_matrix(y_test, y_pred, labels=np.arange(n_bins))
                conf_df = pd.DataFrame(conf_mat, 
                    index=[f'True {i}' for i in range(n_bins)],
                    columns=[f'Pred {i}' for i in range(n_bins)])
                log.write(conf_df.to_string())
                log.write("\n\n")

            # Aggregate fold metrics
            fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
            year_metrics = fold_df.mean().to_dict()
            year_metrics['Year'] = year
            metrics_all_years.append(year_metrics)

    metrics_df = pd.DataFrame(metrics_all_years)
    return metrics_df



def plot_results(metrics_df):
    """
    Plot accuracy, F1 (macro), and recall (macro) over prediction years.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=metrics_df, x='Year', y='Accuracy', label='Accuracy', marker='o')
    sns.lineplot(data=metrics_df, x='Year', y='F1_macro', label='F1 Score (Macro)', marker='s')
    sns.lineplot(data=metrics_df, x='Year', y='Recall_macro', label='Recall (Macro)', marker='^')
    sns.lineplot(data=metrics_df, x='Year', y='Precision_macro', label='Precision (Macro)', marker='x')

    plt.title("Model Classification Performance Over Time (Log-Normal Dist)")
    plt.ylabel("Score")
    plt.xlabel("Prediction Year")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance_df, top_n=10):
    """
    Plot the average feature importance as a horizontal bar chart.
    
    Parameters:
    - importance_df: DataFrame with 'Feature' and 'Importance' columns
    - top_n: number of top features to display
    """
    # Average across folds and years
    mean_importances = (
        importance_df.groupby('Feature')['Importance']
        .mean()
        .sort_values(ascending=True)  # ascending so highest is on top when plotting horizontally
    )

    # Only show top N
    mean_importances = mean_importances.tail(top_n)

    plt.figure(figsize=(10, 0.4 * top_n + 2))
    sns.barplot(x=mean_importances.values, y=mean_importances.index, palette='viridis')
    plt.title(f"Top {top_n} Most Important Features (Avg. Across Folds and Years)")
    plt.xlabel("Average Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.show()

def main():
    # risk_levels_path = 0  # 0 for custom levels, 1 for even levels
    
    # if risk_levels_path == 1:
    #     # Use even levels
    #     mortality_path = even_levels_path
    # else:
    #     # Use custom levels
    #     mortality_path = custom_levels_path
    
    
    df = prepare_yearly_prediction_data_lognormal(lognormal_levels_path)
    
    metrics_df = yearly_classification_prediction_logging(df)

    # Plot results
    plot_results(metrics_df)
    
    # # Plot feature importance
    # plot_feature_importance(importance_df, top_n=14)

if __name__ == "__main__":
    main()



