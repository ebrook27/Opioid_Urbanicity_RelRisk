### 5/21/25, EB: Ok, so in EB_RandomForest_Mortality_Model.py, I used an RF model to predict mortality rates each year,
### using 5-fold CV to get an out of sample prediction for all counties. Then we invesitgated the feature importance,
### to see if county urbanicity category was at all significant. Some years it wasn't super significant, but in a few years 
### it was in the top 5 features. So, following Andrew's plan, I'm going to stratify the counties by urbanicity county first,
### and then use a separate RF model to predict mortality for each urbanicity category.
### Then we can look at the feature importance for each urbanicity category, and hopefully we'll see that the rankings are different
### for each urbanicity category. The plan then would be to get a feature weighting for each category, and then we can use that in a 
### NN/LSTM model to predict mortality rates for each urbanicity category.

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


DATA = ['Mortality', 'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

def prepare_yearly_prediction_data():
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2023))

    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)

    mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
    mort_df['FIPS'] = mort_df['FIPS'].str.zfill(5)
    mort_df = mort_df.set_index('FIPS')

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

    from functools import reduce
    svi_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS', 'year'], how='outer'), svi_data)

    for y in years:
        mort_col = f'{y+1} MR'
        if mort_col not in mort_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'mortality_rate'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(mort_df[mort_col])

    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')
    svi_merged = svi_merged.dropna()

    return svi_merged

# def stratified_rf_by_urbanicity(df, n_splits=5):
#     results = []
#     all_preds = []
#     county_classes = sorted(df['county_class'].unique())

#     for county_class in county_classes:
#         print(f"\n🚩 Running RF for county class: {county_class}")
#         df_sub = df[df['county_class'] == county_class].copy()

#         for year in range(2010, 2022):
#             df_year = df_sub[df_sub['year'] == year].copy()
#             if df_year.empty:
#                 continue

#             X = df_year.drop(columns=['FIPS', 'year', 'mortality_rate', 'county_class'])
#             y = df_year['mortality_rate']
#             fips = df_year['FIPS']

#             kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#             for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
#                 X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#                 y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#                 fips_test = fips.iloc[test_idx]

#                 model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)

#                 mae = mean_absolute_error(y_test, y_pred)
#                 rmse = mean_squared_error(y_test, y_pred, squared=False)
#                 r2 = r2_score(y_test, y_pred)

#                 results.append({
#                     'county_class': county_class,
#                     'year': year,
#                     'fold': fold_idx + 1,
#                     'MAE': mae,
#                     'RMSE': rmse,
#                     'R2': r2
#                 })

#                 fold_preds = pd.DataFrame({
#                     'FIPS': fips_test.values,
#                     'Year': year,
#                     'True': y_test.values,
#                     'Predicted': y_pred,
#                     'county_class': county_class,
#                     'Fold': fold_idx + 1
#                 })
#                 all_preds.append(fold_preds)

#     metrics_df = pd.DataFrame(results)
#     predictions_df = pd.concat(all_preds, ignore_index=True)
#     return metrics_df, predictions_df

CATEGORY_PRED_DIR = 'County Classification/RF_Mort_Preds_by_Urbanicity'
os.makedirs(CATEGORY_PRED_DIR, exist_ok=True)

FEATURE_IMPORTANCE_DIR = 'County Classification/RF_Mort_Preds_by_Urbanicity/RF_Feat_Imp'
os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)

def stratified_rf_by_urbanicity(df, n_splits=5):
    results = []
    all_preds = []
    all_feature_importances = []
    county_classes = sorted(df['county_class'].unique())

    for county_class in county_classes:
        print(f"\n🚩 Running RF for county class: {county_class}")
        df_sub = df[df['county_class'] == county_class].copy()

        for year in range(2010, 2023):
            df_year = df_sub[df_sub['year'] == year].copy()
            if df_year.empty:
                continue

            X = df_year.drop(columns=['FIPS', 'year', 'mortality_rate', 'county_class'])
            y = df_year['mortality_rate']
            fips = df_year['FIPS']

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                fips_test = fips.iloc[test_idx]

                model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)

                results.append({
                    'county_class': county_class,
                    'year': year,
                    'fold': fold_idx + 1,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                })

                fold_preds = pd.DataFrame({
                    'FIPS': fips_test.values,
                    'Year': year,
                    'True': y_test.values,
                    'Predicted': y_pred,
                    'county_class': county_class,
                    'Fold': fold_idx + 1
                })
                all_preds.append(fold_preds)
                
                # Save individual predictions by class and year
                output_path = os.path.join(CATEGORY_PRED_DIR, f"{year}_Cat_{county_class}_MR_predictions.csv")
                fold_preds.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))


                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_,
                    'county_class': county_class,
                    'Year': year,
                    'Fold': fold_idx + 1
                })
                all_feature_importances.append(importance_df)

    metrics_df = pd.DataFrame(results)
    predictions_df = pd.concat(all_preds, ignore_index=True)
    feature_importance_df = pd.concat(all_feature_importances, ignore_index=True)
    return metrics_df, predictions_df, feature_importance_df

# def plot_feature_importance_by_class(feature_importance_df, top_n=15):
#     grouped = (feature_importance_df
#                .groupby(['county_class', 'Feature'], as_index=False)['Importance']
#                .mean())

#     for cclass in grouped['county_class'].unique():
#         subset = grouped[grouped['county_class'] == cclass]
#         top_features = subset.sort_values('Importance', ascending=False).head(top_n)

#         plt.figure(figsize=(10, 6))
#         sns.barplot(data=top_features, x='Importance', y='Feature')
#         plt.title(f"Top {top_n} Features – County Class {cclass}")
#         plt.tight_layout()
#         plt.show()

def plot_feature_importance_by_class(feature_importance_df, top_n=15, save_dir="County Classification/RF_Feat_Imp_Plots/Urbanicity_RF_Feat_Imp_Plots"):
    os.makedirs(save_dir, exist_ok=True)
    grouped = (feature_importance_df
               .groupby(['county_class', 'Feature'], as_index=False)['Importance']
               .mean())

    for cclass in grouped['county_class'].unique():
        subset = grouped[grouped['county_class'] == cclass]
        top_features = subset.sort_values('Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='Importance', y='Feature')
        plt.title(f"Top {top_n} Features – County Class {cclass}")
        plt.tight_layout()

        filename = os.path.join(save_dir, f"Urbanicity_Cat_{cclass}_RF_Feat_Imp.png")
        plt.savefig(filename, dpi=300)
        print(f"✅ Saved: {filename}")
        plt.close()

def main():
    df = prepare_yearly_prediction_data()
    metrics_df, predictions_df, feature_importance_df = stratified_rf_by_urbanicity(df)
    # metrics_df.to_csv("outputs/stratified_rf_metrics.csv", index=False)
    # predictions_df.to_csv("outputs/stratified_rf_predictions.csv", index=False)
    #plot_feature_importance_by_class(feature_importance_df)
    print('Each category`s predictions saved in:', CATEGORY_PRED_DIR)
    
    feature_importance_df.to_csv(os.path.join(FEATURE_IMPORTANCE_DIR, 'RF_feature_importances.csv'), index=False)
    print(f"✅ Saved feature importance rankings to {FEATURE_IMPORTANCE_DIR}/RF_feature_importances.csv")


# Example usage:
if __name__ == "__main__":
    main()