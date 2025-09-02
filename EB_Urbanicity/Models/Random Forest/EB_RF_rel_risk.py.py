### 4/15/25, EB: Ok, once I got the classification actually working, and made sure I wasn't including what I wanted to predict as an input variable (dammit)
### I realized that our approach might not be working. I got terrible, terrible prediction results using the 20 RR levels as the target variable. Even doing 5-fold CV 
### and doing random undersampling on the input, I got accuracy, precision, recall, and F1 scores no higher than 0.3. I think this 

### 4/16/25, EB: From the GridSearchSV tuning, we found that the best parameters for the RF regression model were:
### {'model__max_depth': None, 'model__min_samples_leaf': 2, 'model__min_samples_split': 2, 'model__n_estimators': 250} 

### 4/23/25, EB: I am going to try predicting per-county relative risk scores, rather than sort of binned relative risk score.
### I think this will be a more continuous variable, and should (hopefully) be easier for the RF model to predict.  
### My thinking is that if we can predict these scores, then we can rank them highest to smallest, and then take any high-risk levels we want from that.
### Otherwise, we're having to define our risk levels when we pre-process the data, which feels like bad practice.

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

lognormal_levels_path = 'Data\Mortality\Final Files\Mortality_lognormal_binned_RR.csv'
percentile_path = 'Data\Mortality\Final Files\Mortality_lognormal_percentile_RR.csv'

results_path = 'County Classification\Regression_Preds'
#'County Classification\Regression_Preds'

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

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

    # # Add RR level for year+1
    # for y in years:
    #     rr_col = f'{y+1}_RR_Level'
    #     if rr_col not in rr_df.columns:
    #         continue
    #     svi_merged.loc[svi_merged['year'] == y, 'rr_bin'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(rr_df[rr_col])

    # # Convert rr_bin to integer (from float or object)
    # svi_merged['rr_bin'] = svi_merged['rr_bin'].astype(int)
    
    # Add RR score for year+1
    for y in years:
        rr_score_col = f'{y+1}_RR_Score'
        if rr_score_col not in rr_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'relative_risk_score'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(rr_df[rr_score_col])

    # Add county class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows with missing values
    svi_merged = svi_merged.dropna()

    return svi_merged

def prepare_yearly_prediction_data_percentiles(percentile_path):
    """
    Creates a long-format dataset for predicting next-year mortality percentile ranking
    using current-year SVI + county class.
    """
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2022))  # Use SVI data from 2010–2021 to predict 2011–2022 percentiles

    # Load NCHS urban-rural classification
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

    # Load percentile-ranked mortality data
    percent_df = pd.read_csv(percentile_path, dtype={'FIPS': str})
    percent_df['FIPS'] = percent_df['FIPS'].str.zfill(5)
    percent_df = percent_df.set_index('FIPS')

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
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Add target: next-year percentile
    for y in years:
        perc_col = f'{y+1}_percentile'
        if perc_col not in percent_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'percentile'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(percent_df[perc_col])

    # Add county class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows with missing values (e.g., SVI, percentile, or class)
    svi_merged = svi_merged.dropna()

    return svi_merged

### 4/23/25, EB: This function will take in the per-county RR data in the file Data\Mortality\Final Files\Mortality_RR_per_county.csv
def prepare_yearly_prediction_data_rr(mortality_path):
    """Creates a long-format dataset for predicting next-year relative risk using current-year SVI + county class."""

    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2022))  # Predict RR in years 2011–2022 based on year 2010–2021 SVI

    # Load county classification
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

    # Load relative risk data (already in long format)
    rr_df = pd.read_csv(mortality_path, dtype={'FIPS': str})
    rr_df['FIPS'] = rr_df['FIPS'].str.zfill(5)

    # Rename the column to clarify it's the future RR
    rr_df = rr_df.rename(columns={'year': 'future_year', 'relative_risk': 'relative_risk_score'})

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

    # Merge all SVI variables into one DataFrame
    from functools import reduce
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Merge with RR data — aligning RR for year t+1
    svi_merged['future_year'] = svi_merged['year'] + 1
    merged = svi_merged.merge(rr_df[['FIPS', 'future_year', 'relative_risk_score']],
                              on=['FIPS', 'future_year'], how='left')

    # Merge in county class
    merged = merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop missing values
    merged = merged.dropna()

    return merged



def yearly_rr_regression(df, n_splits=5):
    """
    Predicts continuous relative risk scores using Random Forest regression,
    with cross-validation across counties within each year.
    This now also collects feature importance rankings, averaged across folds, along with absolute errors,
    to plot in a histogram later.
    """
    metrics_all_years = []
    feature_importance_all = []
    all_errors = []

    for year in range(2010, 2022):
        print(f"\n🔁 Processing year {year} → predicting RR score for year {year+1}")
        df_year = df[df['year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            continue

        feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'rr_bin', 'relative_risk_score']]
        X = df_year[feature_cols]
        y = df_year['relative_risk_score']

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['county_class'])
            ], remainder='passthrough')

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('model', RandomForestRegressor(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            abs_errors = np.abs(y_test - y_pred)
            all_errors.extend(abs_errors.tolist())  # Collect all absolute errors

            fold_metrics.append({
                'fold': fold_idx+1,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
            
            importances = pipeline.named_steps['model'].feature_importances_
            feature_names = pipeline.named_steps['prep'].get_feature_names_out()
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Year': year,
                'Fold': fold_idx + 1
            })
            feature_importance_all.append(feature_df)

        # Average across folds
        fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
        year_metrics = fold_df.mean().to_dict()
        year_metrics['Year'] = year
        metrics_all_years.append(year_metrics)
        
        feature_importance_df = pd.concat(feature_importance_all, ignore_index=True)

    return pd.DataFrame(metrics_all_years), feature_importance_df, all_errors

def yearly_rr_regression_with_tuning(df, n_splits=5):
    """
    Predicts continuous RR scores using RF regression with per-year hyperparameter tuning.
    """
    from sklearn.model_selection import KFold

    metrics_all_years = []
    feature_importance_all = []

    param_grid = {
        'model__n_estimators': [100, 250],
        'model__max_depth': [10, 15, None],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }

    for year in range(2010, 2021):
        print(f"\n🔁 Processing year {year} → predicting RR score for year {year+1}")
        df_year = df[df['year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            continue

        feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'rr_bin', 'relative_risk_score']]
        X = df_year[feature_cols]
        y = df_year['relative_risk_score']

        # Preprocessing
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), ['county_class'])
        ], remainder='passthrough')

        base_pipeline = Pipeline([
            ('prep', preprocessor),
            ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])

        # Grid search for best model on the whole year data
        search = GridSearchCV(base_pipeline, param_grid, scoring='r2', cv=3, n_jobs=-1)
        search.fit(X, y)

        print(f"✅ Best params for year {year}: {search.best_params_}")

        # Cross-validation with best model
        best_pipeline = search.best_estimator_

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            best_pipeline.fit(X_train, y_train)
            y_pred = best_pipeline.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            fold_metrics.append({
                'fold': fold_idx+1,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

            # Feature importance
            importances = best_pipeline.named_steps['model'].feature_importances_
            feature_names = best_pipeline.named_steps['prep'].get_feature_names_out()
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Year': year,
                'Fold': fold_idx + 1
            })
            feature_importance_all.append(feature_df)

        fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
        year_metrics = fold_df.mean().to_dict()
        year_metrics['Year'] = year
        metrics_all_years.append(year_metrics)

    metrics_df = pd.DataFrame(metrics_all_years)
    feature_importance_df = pd.concat(feature_importance_all, ignore_index=True)

    return metrics_df, feature_importance_df

def yearly_rr_regression_save_results(df, n_splits=5, save_dir=results_path):
    """
    Predicts continuous relative risk scores using Random Forest regression,
    with cross-validation across counties within each year.
    Saves predictions per year to CSV for mapping.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    metrics_all_years = []
    feature_importance_all = []

    for year in range(2010, 2021):
        print(f"\n🔁 Processing year {year} → predicting RR score for year {year+1}")
        df_year = df[df['year'] == year].copy()
        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            continue

        feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'rr_bin', 'relative_risk_score']]
        X = df_year[feature_cols]
        y = df_year['relative_risk_score']
        fips = df_year['FIPS'].reset_index(drop=True)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        prediction_records = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            fips_test = fips.iloc[test_idx]

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['county_class'])
            ], remainder='passthrough')

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('model', RandomForestRegressor(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Store predictions for this fold
            fold_pred_df = pd.DataFrame({
                'FIPS': fips_test,
                'True_RR': y_test.values,
                'Pred_RR': y_pred
            })
            prediction_records.append(fold_pred_df)

            # Metrics
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            fold_metrics.append({'fold': fold_idx+1, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

            # Feature importances
            importances = pipeline.named_steps['model'].feature_importances_
            feature_names = pipeline.named_steps['prep'].get_feature_names_out()
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Year': year,
                'Fold': fold_idx + 1
            })
            feature_importance_all.append(feature_df)

        # Average fold metrics
        fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
        year_metrics = fold_df.mean().to_dict()
        year_metrics['Year'] = year
        metrics_all_years.append(year_metrics)

        # Combine prediction records and average if needed
        all_preds_df = pd.concat(prediction_records)
        avg_preds_df = all_preds_df.groupby('FIPS').agg({
            'True_RR': 'mean',
            'Pred_RR': 'mean'
        }).reset_index()

        # Rename for clarity when saving
        avg_preds_df.rename(columns={
            'True_RR': f'{year+1}_True_RR',
            'Pred_RR': f'{year+1}_Pred_RR'
        }, inplace=True)

        # Save to CSV
        save_path = os.path.join(save_dir, f'{year+1}_rr_predictions.csv')
        avg_preds_df.to_csv(save_path, index=False)
        print(f"💾 Saved predictions to: {save_path}")

    feature_importance_df = pd.concat(feature_importance_all, ignore_index=True)
    return pd.DataFrame(metrics_all_years), feature_importance_df

### 4/23/25, EB: This function will train and predict on the per-county RR data in the file Data\Mortality\Final Files\Mortality_RR_per_county.csv
def yearly_rr_regression_percounty_RR(df, n_splits=5):
    """
    This function is designed to be used with the per-county RR data.
    Predicts continuous relative risk scores using Random Forest regression,
    with 5-fold cross-validation across counties within each year.
    Also returns feature importances and absolute prediction errors.
    """
    metrics_all_years = []
    feature_importance_all = []
    all_errors = []
    #error_by_percentile = []
    cv_predictions = []
    
    for year in range(2010, 2022):
        print(f"\n🔁 Processing year {year} → predicting RR score for year {year+1}")
        df_year = df[df['year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            continue

        # Drop only columns we don't want as features
        feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'relative_risk_score', 'future_year']]
        X = df_year[feature_cols]
        y = df_year['relative_risk_score']

        # Ensure county_class is treated as categorical
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['county_class'])
            ], remainder='passthrough')

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('model', RandomForestRegressor(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # # Store errors for top risk tiers
            # percentile_results = compute_percentile_errors(
            #     y_test=y_test.values,
            #     y_pred=y_pred,
            #     fips_series=df_year.iloc[test_idx]['FIPS'],
            #     year=year
            # )
            # error_by_percentile.extend(percentile_results)

            cv_predictions.append(pd.DataFrame({
                'FIPS': df_year.iloc[test_idx]['FIPS'].values,
                'year': year,
                'y_true': y_test.values,
                'y_pred': y_pred
            }))


            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            abs_errors = np.abs(y_test - y_pred)
            all_errors.extend(abs_errors.tolist())  # Collect all absolute errors

            fold_metrics.append({
                'fold': fold_idx+1,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

            # Extract and record feature importances
            importances = pipeline.named_steps['model'].feature_importances_
            feature_names = pipeline.named_steps['prep'].get_feature_names_out()
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Year': year,
                'Fold': fold_idx + 1
            })
            feature_importance_all.append(feature_df)

        # Aggregate performance metrics for the year
        fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
        year_metrics = fold_df.mean().to_dict()
        year_metrics['Year'] = year
        metrics_all_years.append(year_metrics)
        df_preds = pd.concat(cv_predictions, ignore_index=True)
        df_preds['abs_error'] = np.abs(df_preds['y_true'] - df_preds['y_pred'])


    # Combine feature importances across all folds and years
    feature_importance_df = pd.concat(feature_importance_all, ignore_index=True)

    #return pd.DataFrame(metrics_all_years), feature_importance_df, all_errors
    return pd.DataFrame(metrics_all_years), feature_importance_df, all_errors, df_preds #pd.DataFrame(error_by_percentile)


def compute_percentile_errors(y_test, y_pred, fips_series, year, percentiles=[0.001, 0.005, 0.01, 0.05]):
    """
    Compute MAE and MSE for different top-percentile slices of predicted RR.
    Returns a list of dicts with results for each slice.
    """
    df_eval = pd.DataFrame({
        'FIPS': fips_series,
        'year': year,
        'y_true': y_test,
        'y_pred': y_pred
    }).reset_index(drop=True)

    # Compute predicted percentiles
    df_eval['pred_rank'] = df_eval['y_pred'].rank(pct=True, ascending=False)

    results = []
    for p in percentiles:
        subset = df_eval[df_eval['pred_rank'] <= p]
        if subset.empty:
            continue
        mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
        mse = mean_squared_error(subset['y_true'], subset['y_pred'])
        results.append({
            'Year': year,
            'Percentile': p,
            'Top_N_Counties': len(subset),
            'MAE': mae,
            'MSE': mse
        })
    
    return results

def plot_percentile_errors(df_percentile_errors):
    """
    Plots MAE and MSE across years for different top percentile groups.
    """
    # Convert percentiles to readable strings for plotting
    df = df_percentile_errors.copy()
    df['Percentile Label'] = (df['Percentile'] * 100).astype(str) + '%'

    # Sort for nice plotting
    df = df.sort_values(['Percentile', 'Year'])

    # === Plot MAE ===
    plt.figure(figsize=(12, 6))
    for label, group in df.groupby('Percentile Label'):
        plt.plot(group['Year'], group['MAE'], label=f'Top {label}')
    plt.title('Mean Absolute Error by Risk Tier Over Time')
    plt.xlabel('Year')
    plt.ylabel('MAE')
    plt.legend(title="Percentile")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot MSE ===
    plt.figure(figsize=(12, 6))
    for label, group in df.groupby('Percentile Label'):
        plt.plot(group['Year'], group['MSE'], label=f'Top {label}')
    plt.title('Mean Squared Error by Risk Tier Over Time')
    plt.xlabel('Year')
    plt.ylabel('MSE')
    plt.legend(title="Percentile")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def assign_risk_levels(df_pred):
    # Load RR data to get true RR scores
    rr_df = pd.read_csv("Data\Mortality\Final Files\Mortality_RR_per_county.csv", dtype={'FIPS': str})
    rr_df['FIPS'] = rr_df['FIPS'].str.zfill(5)

    # Merge true RR into prediction DataFrame
    df_preds = df_preds.merge(rr_df[['FIPS', 'year', 'relative_risk']], on=['FIPS', 'year'], how='left')

    # Rank by RR per year
    df_preds['rr_rank'] = df_preds.groupby('year')['relative_risk'].rank(pct=True, ascending=False)

    # Assign risk level bins
    def assign_risk_bin(p):
        if p <= 0.001:
            return 'Top 0.1%'
        elif p <= 0.005:
            return 'Top 0.5%'
        elif p <= 0.01:
            return 'Top 1%'
        elif p <= 0.05:
            return 'Top 5%'
        elif p <= 0.10:
            return 'Top 10%'
        else:
            return 'Bottom 90%'

    df_preds['risk_level'] = df_preds['rr_rank'].apply(assign_risk_bin)
    
    return df_preds

def plot_error_histograms_by_risk(df_preds):
    """
    Plots histograms of absolute prediction error grouped by risk level.
    """
    plt.figure(figsize=(12, 8))
    risk_levels = ['Top 0.1%', 'Top 0.5%', 'Top 1%', 'Top 5%', 'Top 10%', 'Bottom 90%']
    
    for i, risk in enumerate(risk_levels, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df_preds[df_preds['risk_level'] == risk]['abs_error'], bins=30, kde=True)
        plt.title(f'Absolute Error: {risk}')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.suptitle('Error Distribution by Risk Level', fontsize=16, y=1.02)
    plt.show()


def plot_metrics(metrics_df, all_errors=None):
    
    # We first plot a histogram of the abs. errors for each county.
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=40, edgecolor='black', alpha=0.7)
    plt.title("Histogram of Absolute Errors Across All Years")
    plt.xlabel("Absolute Error (|True RR - Predicted RR|)")
    plt.ylabel("Number of Counties")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # Next we plot a line plot of the metrics over time.
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=metrics_df, x='Year', y='RMSE', label='RMSE', marker='o')
    sns.lineplot(data=metrics_df, x='Year', y='MAE', label='MAE', marker='s')
    sns.lineplot(data=metrics_df, x='Year', y='R2', label='R² Score', marker='^')

    plt.title("Regression Model Error Metrics Over Time")
    plt.ylabel("Metric Value")
    plt.xlabel("Prediction Year")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted_rr(results_dict, year):
    """
    Plots actual vs predicted RR scores for a given year.

    Parameters:
        results_dict (dict): Dictionary returned from regression function, containing predictions.
        year (int): Year to plot (e.g., 2015)
    """
    if year not in results_dict:
        print(f"❌ Year {year} not found in results.")
        return

    data = results_dict[year]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='actual', y='predicted', data=data, alpha=0.4)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfect Prediction')
    plt.xlabel("Actual RR Score")
    plt.ylabel("Predicted RR Score")
    plt.title(f"📈 Actual vs Predicted RR Scores — Year {year}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_regression_feature_importance(feature_importance_df, top_n=15):
    """
    Plots average feature importance across all years/folds.
    """
    avg_importance = (
        feature_importance_df
        .groupby('Feature')['Importance']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_importance.values, y=avg_importance.index, palette='viridis')
    plt.xlabel("Mean Importance (across years/folds)")
    plt.title(f"Top {top_n} Most Important Features — Regression Model")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
#     # Prepare data
#     mortality_path = lognormal_levels_path
#     df = prepare_yearly_prediction_data_lognormal(mortality_path)
# #    print(df.head())
#     print(df['year'].max())

#     # Run regression analysis
#     metrics_df, feature_importance_df, all_errors = yearly_rr_regression(df, n_splits=5)#, save_dir=results_path)

#     # Plot Results
#     plot_metrics(metrics_df, all_errors)
#     #plot_regression_feature_importance(feature_importance_df, top_n=16)
#     #plot_actual_vs_predicted_rr(metrics_df, year=2020)
    
#     # Save results
#     #metrics_df.to_csv('Data/Mortality/Final Files/Yearly_RR_regression_metrics.csv', index=False)
#     #print("✅ Regression metrics saved to: Yearly_RR_regression_metrics.csv")
#     print(metrics_df)
    
    
    ### 4/23/25, EB: Here is a sequence that will take the per-county RR data and run the regression model on it.
    mortality_path = 'Data/Mortality/Final Files/Mortality_RR_per_county.csv'
    df = prepare_yearly_prediction_data_rr(mortality_path)
    
    metrics_df, feature_importance_df, all_errors, df_preds = yearly_rr_regression_percounty_RR(df, n_splits=5)
    df_preds = assign_risk_levels(df_preds)
    plot_error_histograms_by_risk(df_preds)
    plot_metrics(metrics_df, all_errors)
    #plot_percentile_errors(df_percentile_errors)


if __name__ == "__main__":
    main()