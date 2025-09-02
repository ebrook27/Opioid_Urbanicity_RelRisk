### 4/21/25, EB: Here I am trying to use an RF regression model to predict the log-transformed mortality rates.
### I think that instead of predicting the risk directly, we should predict the mortality rates, then transform them to risk.
### The log-transform should (hopefully) help the model predict better.

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

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

def prepare_yearly_prediction_data_log_mortality():
    """
    Prepares a long-format dataset for predicting next-year log(Mortality Rate)
    using current-year SVI + county class as inputs.
    """
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2022))  # Predict for 2011–2022 using 2010–2021 data

    # Load county urbanicity class
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

    # Load mortality data
    mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
    mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
    mort_df = mort_df.set_index('FIPS')

    # Load and reshape SVI variables
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

    # Merge all SVI variables into one long dataframe
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    # Add next-year log(Mortality Rate)
    for y in years:
        mr_col = f'{y+1} MR'
        if mr_col not in mort_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'log_mortality_next'] = svi_merged.loc[
            svi_merged['year'] == y, 'FIPS'].map(mort_df[mr_col]).apply(lambda x: np.log1p(x) if pd.notnull(x) else None)

    # Add urbanicity class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows with missing data
    svi_merged = svi_merged.dropna()

    return svi_merged



def yearly_log_mortality_regression(df, n_splits=5):
    """
    Predicts log-transformed mortality rates using Random Forest regression,
    with cross-validation within each year.
    Returns yearly metrics, feature importances, and all absolute prediction errors.
    """
    metrics_all_years = []
    feature_importance_all = []
    all_errors = []

    for year in range(2010, 2022):
        print(f"\n🔁 Processing year {year} → predicting log(Mortality Rate) for year {year+1}")
        df_year = df[df['year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            continue

        feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'log_mortality_next']]
        X = df_year[feature_cols]
        y = df_year['log_mortality_next']

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
                'fold': fold_idx + 1,
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

def stratified_log_mortality_regression(df, n_splits=5):
    """
    Trains separate Random Forest regressors on pre-COVID and post-COVID years.
    Returns metrics, feature importances, and error distributions for each period.
    """
    def run_pipeline(year_range):
        metrics_all = []
        feature_importance_all = []
        all_errors = []

        for year in year_range:
            print(f"\n🔁 Processing year {year} → predicting log(Mortality Rate) for year {year+1}")
            df_year = df[df['year'] == year].copy()

            if df_year.empty:
                print(f"⚠️ Skipping year {year}: no data.")
                continue

            feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'log_mortality_next']]
            X = df_year[feature_cols]
            y = df_year['log_mortality_next']

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
                all_errors.extend(abs_errors.tolist())

                fold_metrics.append({
                    'fold': fold_idx + 1,
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

            # Average metrics across folds
            fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
            year_metrics = fold_df.mean().to_dict()
            year_metrics['Year'] = year
            metrics_all.append(year_metrics)

        feature_importance_df = pd.concat(feature_importance_all, ignore_index=True)
        return pd.DataFrame(metrics_all), feature_importance_df, all_errors

    # Stratified by COVID: pre vs post
    pre_covid_years = list(range(2010, 2020))  # predicting 2011–2020
    post_covid_years = list(range(2020, 2022))  # predicting 2021–2022

    print("\n📊 Running pre-COVID model analysis...")
    pre_metrics, pre_importance, pre_errors = run_pipeline(pre_covid_years)

    print("\n📊 Running post-COVID model analysis...")
    post_metrics, post_importance, post_errors = run_pipeline(post_covid_years)

    return {
        'pre': {
            'metrics': pre_metrics,
            'importances': pre_importance,
            'errors': pre_errors
        },
        'post': {
            'metrics': post_metrics,
            'importances': post_importance,
            'errors': post_errors
        }
    }



def plot_metrics(metrics_df, covid_tag, all_errors=None):
    
    # We first plot a histogram of the abs. errors for each county.
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=40, edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of Absolute Errors Across All Years, {covid_tag}")
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

    plt.title(f"Regression Model Error Metrics Over Time, {covid_tag}")
    plt.ylabel("Metric Value")
    plt.xlabel("Prediction Year")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_regression_feature_importance(feature_importance_df, covid_tag, top_n=15):
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
    plt.title(f"Top {top_n} Most Important Features — Regression Model, {covid_tag}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def main():
    #########################################################################################################
    ### 4/22/25, EB: Here we are implementing a regression analysis using the log-transformed mortality rates, for all years in the study.
    
    # print("Using SVI data to predict log-transformed mortality rates...")
    # # Prepare data
    # df = prepare_yearly_prediction_data_log_mortality()
    # #print(df['relative_risk_score'].head())

    # # Run regression analysis
    # metrics_df, feature_importance_df, all_errors = yearly_log_mortality_regression(df, n_splits=5)#, save_dir=results_path)
    

    # # Plot Results
    # plot_metrics(metrics_df, all_errors)
    # plot_regression_feature_importance(feature_importance_df, top_n=16)
    # #plot_actual_vs_predicted_rr(metrics_df, year=2020)
    
    # print("Regression analysis complete.")
    
    # # Save results
    # #metrics_df.to_csv('Data/Mortality/Final Files/Yearly_RR_regression_metrics.csv', index=False)
    # #print("✅ Regression metrics saved to: Yearly_RR_regression_metrics.csv")
    # #print(metrics_df)

    #########################################################################################################
    ### 4/22/25, EB: Here we are implementing a pre/post-COVID stratified analysis.
    
    print('----------------------------------------')
    print("Using SVI data to predict log-transformed mortality rates, Pre-COVID vs Post-COVID...")
    # Prepare data
    df = prepare_yearly_prediction_data_log_mortality()
    
    # Run the stratified regression analysis
    results_dict = stratified_log_mortality_regression(df, n_splits=5)
    
    #### Error metrics for pre-COVID and post-COVID
    # Pre-COVID
    print("📉 Pre-COVID error and metrics:")
    plot_metrics(results_dict['pre']['metrics'], covid_tag='Pre-Covid', all_errors=results_dict['pre']['errors'])

    # Post-COVID
    print("📉 Post-COVID error and metrics:")
    plot_metrics(results_dict['post']['metrics'], covid_tag='Post-Covid', all_errors=results_dict['post']['errors'])

    #### Feature importance for pre-COVID and post-COVID
    # Pre-COVID feature importance
    print("🌲 Pre-COVID feature importance:")
    plot_regression_feature_importance(results_dict['pre']['importances'], covid_tag='Pre-Covid')

    # Post-COVID feature importance
    print("🌲 Post-COVID feature importance:")
    plot_regression_feature_importance(results_dict['post']['importances'], covid_tag='Post-Covid')
    
    print("Stratified regression analysis complete.")
    print('----------------------------------------')



if __name__ == "__main__":
    main()