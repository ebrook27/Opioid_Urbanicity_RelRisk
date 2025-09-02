### 5/20/25, EB: Alright. We've tried a ton of different things, and nothing's really sticking. I talked with Andrew today and he gave me a suggestion for a sort of pipeline to try.
### We're going to start by a vanilla Random Forest model, including urbanicity category as input, and when we look at feature importance, hopefully we see that it's at least somewhat important.
### If that works, then we can try using a separate RF model for each urbanicity category, and see how well we can predict mortality within that way.
### Ideally, THEN we can weight the features by importance for each category, and use that as input to some NN/LSTM model. One step at a time.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


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

def load_prediction_data(df):
    """Prepares the data for training and testing the Random Forest model."""
    # Prepare the data
    feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_rate', 'county_class']]
    X = df[feature_cols + ['county_class']].copy()
    y = df['mortality_rate'].copy()

    # One-hot encode urbanicity
    X = pd.get_dummies(X, columns=['county_class'], drop_first=True)
    
    return X, y


def train_random_forest(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, preds):.4f}")
    print(f"Test R² : {r2_score(y_test, preds):.4f}")

    return model, X.columns

def cross_validated_predictions(X, y, n_splits=5, random_state=42):
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Generate out-of-sample predictions for every row
    preds = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)

    # Evaluate
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"Cross-Validated MSE: {mse:.4f}")
    print(f"Cross-Validated R² : {r2:.4f}")

    # Fit on full data (optional) for feature importance
    model.fit(X, y)

    return preds, model, X.columns

### 5/20/25, EB: above cross_validated_predictions() function ran remakrably slow, so I had Chat refactor it.
### I think the cross_val+predict() function was slowing things down, so I just used KFold() to create the splits and then fit/predict on each fold.


def yearly_mortality_prediction(df, n_splits=5):
    """
    Predicts next-year opioid mortality using Random Forest regression,
    with k-fold cross-validation across counties within each year.
    Returns metrics, feature importances, and per-sample errors.
    """
    metrics_all_years = []
    feature_importance_all = []
    all_errors = []
    all_predictions = []

    for year in range(2010, 2023):
        print(f"\n🔁 Processing year {year} → predicting mortality for year {year+1}")
        df_year = df[df['year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            continue

        feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_rate']]
        X = df_year[feature_cols]
        y = df_year['mortality_rate']
        fips = df_year['FIPS']

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

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

            # Collect errors
            abs_errors = np.abs(y_test - y_pred)
            all_errors.extend(abs_errors.tolist())

            # Save per-county predictions
            fold_df = pd.DataFrame({
                'FIPS': fips_test.values,
                'Year': year,
                'True': y_test.values,
                'Predicted': y_pred,
                'Fold': fold_idx + 1
            })
            all_predictions.append(fold_df)

            # Fold metrics
            fold_metrics.append({
                'fold': fold_idx + 1,
                'RMSE': mean_squared_error(y_test, y_pred, squared=False),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            })

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

        # Average metrics across folds for this year
        fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
        year_metrics = fold_df.mean().to_dict()
        year_metrics['Year'] = year
        metrics_all_years.append(year_metrics)

    # Combine all results
    metrics_df = pd.DataFrame(metrics_all_years)
    feature_importance_df = pd.concat(feature_importance_all, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return metrics_df, feature_importance_df, predictions_df, all_errors


def get_top_feature_importance_for_year(feature_df, year, top_n=20):
    df_year = feature_df[feature_df['Year'] == year]
    agg_df = df_year.groupby('Feature', as_index=False)['Importance'].mean()
    top_df = agg_df.sort_values('Importance', ascending=False).head(top_n)
    return top_df

def plot_feature_importance(top_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_df, x='Importance', y='Feature')
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

def plot_feature_importance_all_years(feature_df, start_year=2010, end_year=2021, top_n=20):
    for year in range(start_year, end_year + 1):
        df_year = feature_df[feature_df['Year'] == year]

        if df_year.empty:
            print(f"⚠️ No feature importance data for year {year}, skipping.")
            continue

        agg_df = df_year.groupby('Feature', as_index=False)['Importance'].mean()
        top_df = agg_df.sort_values('Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_df, x='Importance', y='Feature')
        plt.title(f"Top {top_n} Feature Importances – Year {year}")
        plt.tight_layout()
        plt.show()

# def plot_grouped_feature_importance_all_years(feature_df, start_year=2010, end_year=2021, top_n=20):
#     for year in range(start_year, end_year + 1):
#         df_year = feature_df[feature_df['Year'] == year].copy()

#         if df_year.empty:
#             print(f"⚠️ No feature importance data for year {year}, skipping.")
#             continue

#         # Step 1: Average importances across folds
#         df_year = df_year.groupby('Feature', as_index=False)['Importance'].mean()

#         # Step 2: Map one-hot encoded components to their original feature
#         def group_feature(name):
#             if name.startswith('county_class'):
#                 return 'County Category'
#             else:
#                 return name

#         df_year['GroupedFeature'] = df_year['Feature'].apply(group_feature)

#         # Step 3: Aggregate importance by grouped feature
#         grouped_df = df_year.groupby('GroupedFeature', as_index=False)['Importance'].sum()
#         top_df = grouped_df.sort_values('Importance', ascending=False).head(top_n)

#         # Step 4: Plot
#         plt.figure(figsize=(10, 6))
#         sns.barplot(data=top_df, x='Importance', y='GroupedFeature')
#         plt.title(f"Grouped Feature Importance – Year {year}")
#         plt.tight_layout()
#         plt.show()

# def plot_grouped_feature_importance_all_years(feature_df, start_year=2010, end_year=2021, top_n=20):
#     for year in range(start_year, end_year + 1):
#         df_year = feature_df[feature_df['Year'] == year].copy()

#         if df_year.empty:
#             print(f"⚠️ No feature importance data for year {year}, skipping.")
#             continue

#         # Step 1: Average importances across folds
#         df_year = df_year.groupby('Feature', as_index=False)['Importance'].mean()

#         # Step 2: Map one-hot encoded components to their original feature
#         def group_feature(name):
#             if 'county_class' in name:
#                 return 'County Category'
#             else:
#                 return name

#         df_year['GroupedFeature'] = df_year['Feature'].apply(group_feature)

#         # Step 3: Aggregate importance by grouped feature
#         grouped_df = df_year.groupby('GroupedFeature', as_index=False)['Importance'].sum()
#         top_df = grouped_df.sort_values('Importance', ascending=False).head(top_n)

#         # Step 4: Plot
#         plt.figure(figsize=(10, 6))
#         sns.barplot(data=top_df, x='Importance', y='GroupedFeature')
#         plt.title(f"Grouped Feature Importance – Year {year}")
#         plt.tight_layout()
#         plt.show()

import os

def plot_grouped_feature_importance_all_years(feature_df, start_year=2010, end_year=2021, top_n=20, save_dir="County Classification\RF_Feat_Imp_Plots", show_plots=False):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        df_year = feature_df[feature_df['Year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ No feature importance data for year {year}, skipping.")
            continue

        # # Step 1: Average importances across folds
        # df_year = df_year.groupby('Feature', as_index=False)['Importance'].mean()

        # # Step 2: Map one-hot encoded components to their original feature
        # def group_feature(name):
        #     if 'county_class' in name:
        #         return 'County Category'
        #     else:
        #         return name

        # df_year['GroupedFeature'] = df_year['Feature'].apply(group_feature)
        
        # Remove prefixes like "remainder__" or "prep__cat__"
        df_year['Feature'] = df_year['Feature'].str.replace(r'^(remainder__|prep__cat__)', '', regex=True)

        # Then apply grouping
        def group_feature(name):
            if 'county_class' in name:
                return 'County Category'
            else:
                return name

        df_year['GroupedFeature'] = df_year['Feature'].apply(group_feature)

        # Step 3: Aggregate importance by grouped feature
        grouped_df = df_year.groupby('GroupedFeature', as_index=False)['Importance'].sum()
        top_df = grouped_df.sort_values('Importance', ascending=False).head(top_n)

        # Step 4: Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_df, x='Importance', y='GroupedFeature')
        plt.title(f"Grouped Feature Importance – Year {year}")
        plt.tight_layout()

        # # Save figure
        # filename = os.path.join(save_dir, f"RF_feat_imp_{year}.png")
        # plt.savefig(filename, dpi=300)
        # print(f"✅ Saved: {filename}")

        # Optionally show the plot
        if show_plots:
            plt.show()

        plt.close()  # Avoid overlapping figures if looping


# def plot_grouped_feature_importance_summary(feature_df, start_year=2010, end_year=2021, top_n=20, save_path="County Classification/RF_Feat_Imp_Plots/RF_feat_imp_summary.png"):
#     """
#     Plots the feature importance summary for the RF's predictions across all years. Ranks the features by their average importance across years,
#     and shows the importance for each year in a horizontal bar plot. Similar to Andrew's plot from the XGB paper.
#     """
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # Clean and group feature names
#     feature_df['Feature'] = feature_df['Feature'].str.replace(r'^(remainder__|prep__cat__)', '', regex=True)

#     def group_feature(name):
#         if 'county_class' in name:
#             return 'County Category'
#         else:
#             return name

#     feature_df['GroupedFeature'] = feature_df['Feature'].apply(group_feature)

#     # Compute mean importance across folds for each year-feature combination
#     year_avg = feature_df.groupby(['Year', 'GroupedFeature'], as_index=False)['Importance'].mean()

#     # Compute overall mean importance across years
#     mean_importance = year_avg.groupby('GroupedFeature', as_index=False)['Importance'].mean()
#     top_features = mean_importance.sort_values('Importance', ascending=False).head(top_n)['GroupedFeature'].tolist()

#     # Filter for top features only
#     filtered = year_avg[year_avg['GroupedFeature'].isin(top_features)]

#     # Pivot for plotting
#     pivot = filtered.pivot(index='GroupedFeature', columns='Year', values='Importance').fillna(0)
#     pivot = pivot.loc[top_features]  # preserve order

#     # Add average column for bar sorting
#     pivot['Average'] = pivot.mean(axis=1)
#     pivot = pivot.sort_values('Average', ascending=False)

#     # Plot
#     plt.figure(figsize=(12, 7))
#     colors = sns.color_palette("tab10", end_year - start_year + 1)
#     pivot.drop(columns='Average').T.plot(kind='barh', stacked=False, ax=plt.gca(), color=colors, legend=False)

#     # Overplot average
#     plt.barh(pivot.index, pivot['Average'], color='black', height=0.4, label='Average')

#     plt.xlabel('Feature Importance (Gain)')
#     plt.title('XGBoost Feature Importance')
#     plt.legend(title='Year', labels=list(range(start_year, end_year + 1)) + ['Average'])
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"✅ Saved summary feature importance plot to {save_path}")

# def plot_grouped_feature_importance_summary(feature_df, start_year=2010, end_year=2021, top_n=20, save_path="County Classification/RF_Feat_Imp_Plots/RF_feat_imp_summary.png"):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # Clean and group feature names
#     feature_df['Feature'] = feature_df['Feature'].str.replace(r'^(remainder__|prep__cat__)', '', regex=True)

#     def group_feature(name):
#         if 'county_class' in name:
#             return 'County Category'
#         else:
#             return name

#     feature_df['GroupedFeature'] = feature_df['Feature'].apply(group_feature)

#     # Compute mean importance across folds for each year-feature combination
#     year_avg = feature_df.groupby(['Year', 'GroupedFeature'], as_index=False)['Importance'].mean()

#     # Compute overall mean importance across years
#     mean_importance = year_avg.groupby('GroupedFeature', as_index=False)['Importance'].mean()
#     top_features = mean_importance.sort_values('Importance', ascending=False).head(top_n)['GroupedFeature'].tolist()

#     # Filter for top features only
#     filtered = year_avg[year_avg['GroupedFeature'].isin(top_features)]

#     # Pivot for plotting
#     pivot = filtered.pivot(index='GroupedFeature', columns='Year', values='Importance').fillna(0)
#     pivot = pivot.loc[top_features]  # preserve order

#     # Plot
#     plt.figure(figsize=(12, 7))
#     colors = sns.color_palette("tab10", end_year - start_year + 1)
#     for year in range(start_year, end_year + 1):
#         if year in pivot.columns:
#             plt.barh(pivot.index, pivot[year], height=0.6, left=None, label=str(year), alpha=0.8)

#     plt.xlabel('Feature Importance (Gain)')
#     plt.title('XGBoost Feature Importance')
#     plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"✅ Saved summary feature importance plot to {save_path}")

def plot_grouped_feature_importance_summary(feature_df, start_year=2010, end_year=2021, top_n=20, save_path="County Classification/RF_Feat_Imp_Plots/RF_feat_imp_summary.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ### 5/27/25, EB: So the following wasn't working, it showed wildly different feature importances for each year, even though the data was the same.
    ### What we should have is average across folds to get each years' importance, then average across years to get the overall importance.
    ### But within the bar for each feature, I want the individual years' importance scores as skinny bars.
    # # Clean and group feature names
    # feature_df['Feature'] = feature_df['Feature'].str.replace(r'^(remainder__|prep__cat__)', '', regex=True)

    # def group_feature(name):
    #     if 'county_class' in name:
    #         return 'County Category'
    #     else:
    #         return name

    # feature_df['GroupedFeature'] = feature_df['Feature'].apply(group_feature)

    # # Compute mean importance across folds for each year-feature combination
    # year_avg = feature_df.groupby(['Year', 'GroupedFeature'], as_index=False)['Importance'].mean()

    ### 5/27/25, EB: The following should do what we want
    
    # Clean and group feature names
    feature_df['Feature'] = feature_df['Feature'].str.replace(r'^(remainder__|prep__cat__)', '', regex=True)

    def group_feature(name):
        if 'county_class' in name:
            return 'County Category'
        else:
            return name

    feature_df['GroupedFeature'] = feature_df['Feature'].apply(group_feature)

    # ⬇️ Group one-hot encoded components before averaging across folds
    fold_grouped = (
        feature_df
        .groupby(['Year', 'Fold', 'GroupedFeature'], as_index=False)['Importance']
        .sum()
    )

    # ⬇️ Then average across folds for each year-feature
    year_avg = (
        fold_grouped
        .groupby(['Year', 'GroupedFeature'], as_index=False)['Importance']
        .mean()
    )


    # Compute overall mean importance across years
    mean_importance = year_avg.groupby('GroupedFeature', as_index=False)['Importance'].mean()
    top_features = mean_importance.sort_values('Importance', ascending=False).head(top_n)['GroupedFeature'].tolist()

    # Filter for top features only
    filtered = year_avg[year_avg['GroupedFeature'].isin(top_features)]

    # Plot
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    
    # Reverse order for highest rank at the top
    feature_order = (
        filtered.groupby("GroupedFeature")['Importance']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Troubleshooting: Ensure 'Year' is a string for consistent plotting
    filtered['Year'] = filtered['Year'].astype(str)
    ax = sns.barplot(
        data=filtered,
        x="Importance",
        y="GroupedFeature",
        hue="Year",
        order=feature_order,
        palette="tab10"
    )

    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_ylabel("")
    ax.set_title("Random Forest Feature Importance across all years")
    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved summary feature importance plot to {save_path}")




def main():
    # df = prepare_yearly_prediction_data()
    # X, y = load_prediction_data(df)
    # model, feature_names = train_random_forest(X, y)
    # plot_feature_importance(model, feature_names)

    # # Optional: Cross-validated predictions
    # df = prepare_yearly_prediction_data()
    # X, y = load_prediction_data(df)
    # preds, model, feature_names = cross_validated_predictions(X, y)
    # plot_feature_importance(model, feature_names)

    df = prepare_yearly_prediction_data()
    metrics, feature_importance, predictions, errors = yearly_mortality_prediction(df)
    # top_2016 = get_top_feature_importance_for_year(feature_importance, year=2016, top_n=20)
    # plot_feature_importance(top_2016)
    #plot_grouped_feature_importance_all_years(feature_importance, show_plots=True)
    plot_grouped_feature_importance_summary(feature_importance, start_year=2010, end_year=2021)




if __name__ == "__main__":
    main()




### The following was from the original code, but I don't think we need it anymore.

# def plot_feature_importance(model, feature_names, top_n=20):
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[-top_n:]

#     plt.figure(figsize=(10, 6))
#     plt.barh(range(top_n), importances[indices])
#     plt.yticks(range(top_n), [feature_names[i] for i in indices])
#     plt.xlabel("Feature Importance")
#     plt.title(f"Top {top_n} Features Predicting Opioid Mortality")
#     plt.tight_layout()
#     plt.show()
