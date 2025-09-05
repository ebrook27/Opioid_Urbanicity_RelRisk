### 8/15/25, EB: In this file and in EB_XGB_Urbanicity_Models.py, we are using the same script as in the EB RandomForest...
### scripts, but we're changing the model to an XGBoost model, rather than a RandomForest model. It was pointed out to me in my 
### oral exam that I hadn't done any work to quantify and account for potential collinearities in the SVI and urbanicity categories,
### so I am now using XGBoost, which is more robust to collinearity than RandomForest.

#### **** #### ****
### 8/21/25, EB: Andrew pointed out to me that we shouldn't sum the feature importances for the county urbanicity categories, we should average them.
### My first thought was each 




import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging



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

# def load_prediction_data(df):
    # """Prepares the data for training and testing the Random Forest model."""
    # # Prepare the data
    # feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_rate', 'county_class']]
    # X = df[feature_cols + ['county_class']].copy()
    # y = df['mortality_rate'].copy()

    # # One-hot encode urbanicity
    # X = pd.get_dummies(X, columns=['county_class'], drop_first=True)
    
    # return X, y


def train_xgb(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, preds):.4f}")
    print(f"Test R² : {r2_score(y_test, preds):.4f}")

    return model, X.columns

def cross_validated_predictions(X, y, n_splits=5, random_state=42):
    model = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
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


### 8/18/25, EB: Updated the following function to use the best XGBoost hyperparameters I found via tuning on 8/18/25.

def yearly_mortality_prediction(df, n_splits=5):
    """
    Predicts next-year opioid mortality using XGBoost regression,
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
                # ('model', xgb.XGBRegressor(
                #     n_estimators=250,
                #     max_depth=None,
                #     min_samples_split=2,
                #     min_samples_leaf=2,
                #     random_state=42,
                #     n_jobs=-1
                # ))
                ('model', xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=7,
                    learning_rate=0.01,
                    subsample=0.7,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    gamma=0,
                    objective='reg:squarederror',
                    n_jobs=-1,
                    random_state=42
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

def yearly_mortality_prediction_native_categorical(df, n_splits=5):
    """
    Same as previous yearly_mortality_prediction() function, but uses XGBoost's native categorical support, rather
    than one-hot encoding the urbanicity categories.
    Predicts next-year opioid mortality using XGBoost regression with native categorical support,
    and k-fold cross-validation across counties within each year.
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
        X = df_year[feature_cols].copy()
        y = df_year['mortality_rate']
        fips = df_year['FIPS']

        # Cast county_class to category dtype for XGBoost
        if 'county_class' in X.columns:
            X['county_class'] = X['county_class'].astype('category')

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            fips_test = fips.iloc[test_idx]

            # Ensure categorical dtype is preserved
            if 'county_class' in X_train.columns:
                X_train['county_class'] = X_train['county_class'].astype('category')
                X_test['county_class'] = X_test['county_class'].astype('category')

            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.01,
                subsample=0.7,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0,
                objective='reg:squarederror',
                # objective='reg:tweedie',
                n_jobs=-1,
                random_state=42,
                enable_categorical=True  # ← key change
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

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
                'RMSE': mean_squared_error(y_test, y_pred),#, squared=False),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            })

            # Feature importances
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': X_train.columns,
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

def plot_feature_importance_each_year(feature_df, start_year=2010, end_year=2021, top_n=20):
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


def plot_grouped_feature_importance_all_years(feature_df, start_year=2010, end_year=2021, top_n=20, save_dir="County Classification\XGB_Feat_Imp_Plots", show_plots=False):
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

        # Save figure
        filename = os.path.join(save_dir, f"XGB_feat_imp_{year}.png")
        plt.savefig(filename, dpi=300)
        print(f"✅ Saved: {filename}")

        # Optionally show the plot
        if show_plots:
            plt.show()

        plt.close()  # Avoid overlapping figures if looping

def plot_grouped_feature_importance_summary(feature_df, start_year=2010, end_year=2021, top_n=20, save_path="County Classification/XGB_Feat_Imp_Plots/XGB_feat_imp_summary.png"):
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
    ax.set_title("XGBoost Feature Importance across all years")
    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved summary feature importance plot to {save_path}")



###############################################################
### 8/21/25, EB: Using the native categorical support in XGBoost to more simply handle the county_class feature, we need to update
### the feature importance plotting functions to account for the fact that the county_class feature is no longer one-hot encoded.

def plot_feature_importance_all_years_native_categorical(
    feature_df,
    start_year=2010,
    end_year=2021,
    top_n=20,
    save_dir="County Classification\XGB_Feat_Imp_Plots",
    show_plots=False
):
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        df_year = feature_df[feature_df['Year'] == year].copy()
        df_year.rename(columns={'county_class': 'Urbanicity Level'}, inplace=True)


        if df_year.empty:
            print(f"⚠️ No feature importance data for year {year}, skipping.")
            continue

        # Step 1: Average feature importances across folds
        mean_importance = (
            df_year.groupby('Feature', as_index=False)['Importance']
            .mean()
            .sort_values('Importance', ascending=False)
        )

        # Step 2: Keep top N
        top_df = mean_importance.head(top_n)

        # Step 3: Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_df, x='Importance', y='Feature')
        plt.title(f"XGBoost Top {top_n} Feature Importances – Year {year}")
        plt.tight_layout()

        # Save plot
        filename = os.path.join(save_dir, f"XGB_feat_imp_{year}_native_categorical_urbanicity_v2.png")
        plt.savefig(filename, dpi=300)
        print(f"✅ Saved: {filename}")

        if show_plots:
            plt.show()

        plt.close()

# def plot_average_feature_importance(feature_importance_df):
#     '''
#     This function is from Andrew's XGBoost model he wrote. It produces the nice big feature importance plot, with the 
#     importances ranked by their average over time, and with a bar for each year within each variable.
#     '''
    
#     # Calculate the average importance across all years
#     feature_importance_df['Average'] = feature_importance_df.mean(axis=1)

#     # Sort the DataFrame by the average importance
#     feature_importance_df = feature_importance_df.sort_values(by='Average', ascending=True)

#     # Get the variables (features) and years (columns)
#     features = feature_importance_df.index
#     years = feature_importance_df.columns

#     # Define bar width and positions for each group
#     bar_width = 0.6
#     y_positions = np.arange(len(features))  # Spacing between feature groups

#     # Define colors
#     num_years = len(years) - 1  # Exclude 'Average'
#     colors = list(plt.cm.tab20.colors[:num_years]) + ['black']  # Add black for 'Average'

#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(12, 8))

#     # Plot each year's bars
#     for i, year in enumerate(years):
#         ax.barh(y_positions - i * bar_width / num_years, feature_importance_df[year],
#                 height=bar_width / num_years, label=year, color=colors[i])

#     # Adjust labels, title, and legend
#     ax.set_yticks(y_positions)
#     ax.set_yticklabels(features, fontsize=20)
#     ax.set_xlabel('Feature Importance (Gain)', fontsize=20, fontweight='bold')
#     ax.tick_params(axis='x', labelsize=20)  # Increase the font size of x-axis tick labels
#     ax.set_title('XGBoost Feature Importance', fontsize=20, fontweight='bold')
#     ax.legend(title='Year', fontsize=15, title_fontsize=15, loc='lower right')

#     # Adjust layout and save
#     plt.tight_layout()
#     #plt.savefig('County Classification/XGB_Feat_Imp_Plots/xgboost_average_feature_importance_all_years.png', bbox_inches='tight')
#     plt.close()

#     # Log the average feature importance for each variable
#     feature_importance_df = feature_importance_df.sort_values(by='Average', ascending=False)
#     logging.info("Average Feature Importance for each variable:")
#     for feature, avg_importance in feature_importance_df['Average'].items():
#         logging.info(f"{feature}: {avg_importance:.4f}")

def plot_average_feature_importance(feature_importance_df):
    """
    This is adapted from Andrew's average feature importance plot function. He used a wide-format df as input,
    and mine was long-format.
    Plots feature importance per year, sorted by average importance across years.
    Expects long-format DataFrame: ['Feature','Importance','Year','Fold'].
    """

    # Pivot: rows = Feature, columns = Year, values = mean importance across folds
    wide_df = (
        feature_importance_df
        .groupby(['Feature','Year'])['Importance'].mean()
        .unstack(fill_value=0)   # Year becomes columns
    )

    # Add average column
    wide_df['Average'] = wide_df.mean(axis=1)

    # Sort by average importance
    wide_df = wide_df.sort_values(by='Average', ascending=True)

    # Features and years
    features = wide_df.index
    years = wide_df.columns

    # Plot parameters
    bar_width = 0.6
    y_positions = np.arange(len(features))
    num_years = len(years) - 1  # exclude 'Average'
    colors = list(plt.cm.tab20.colors[:num_years]) + ['black']

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, year in enumerate(years):
        ax.barh(
            y_positions - i * bar_width / len(years),
            wide_df[year],
            height=bar_width / len(years),
            label=str(year),
            color=colors[i]
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(features, fontsize=14)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=14)
    ax.set_title('XGBoost Feature Importance (per year + avg)', fontsize=18, fontweight='bold')
    ax.legend(title='Year', fontsize=12, title_fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig('County Classification/XGB_Feat_Imp_Plots/xgboost_average_feature_importance_all_years.png', bbox_inches='tight')
    plt.show()

    # Log average importance
    wide_df = wide_df.sort_values(by='Average', ascending=False)
    logging.info("Average Feature Importance for each variable:")
    for feature, avg_importance in wide_df['Average'].items():
        logging.info(f"{feature}: {avg_importance:.4f}")



### 8/18/25, EB: Adding a gridsearch hyperparameter tuning function for XGBoost
def tune_xgb_hyperparameters_by_year(df, n_splits=5, n_iter=30, random_state=42, output_dir=None):
    """
    Performs RandomizedSearchCV hyperparameter tuning for XGBRegressor by year.
    Returns a DataFrame of best parameters and scores for each year.
    Optionally saves full CV results if output_dir is given.
    """
    all_best_params = []
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    for year in range(2010, 2023):
        print(f"\n🔧 Hyperparameter tuning for year {year}")
        df_year = df[df['year'] == year].copy()
        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            continue

        X = df_year.drop(columns=['FIPS', 'year', 'mortality_rate'])
        y = df_year['mortality_rate']

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), ['county_class'])
        ], remainder='passthrough')

        pipeline = Pipeline([
            ('prep', preprocessor),
            ('model', xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state, n_jobs=-1))
        ])

        param_dist = {
            'model__n_estimators': [100, 250, 500],
            'model__max_depth': [3, 5, 7, 10],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample': [0.7, 0.9, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__min_child_weight': [1, 5, 10],
            'model__gamma': [0, 1, 5]
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            cv=kf,
            verbose=2,
            random_state=random_state,
            n_jobs=-1,
            return_train_score=True
        )

        search.fit(X, y)

        # Save best result
        best = {
            'Year': year,
            'Best RMSE': -search.best_score_,
            **search.best_params_
        }
        all_best_params.append(best)

        # Save all results if needed
        if output_dir:
            cv_results_df = pd.DataFrame(search.cv_results_)
            cv_results_df.to_csv(os.path.join(output_dir, f"xgb_cv_results_{year}.csv"), index=False)

    return pd.DataFrame(all_best_params)



#############
### 9/4/25, EB: Adding a function that plots the absolute error histograms for each year
def plot_abs_error_histograms(predictions_df, save_plots=False, output_dir="./evaluation_plots"):
    """
    Plot separate histograms of absolute prediction error for each year.
    predictions_df must have columns: ["Year", "True", "Predicted"].
    """

    # compute absolute error
    predictions_df = predictions_df.copy()
    predictions_df["AbsError"] = np.abs(predictions_df["True"] - predictions_df["Predicted"])

    unique_years = sorted(predictions_df["Year"].unique())

    for year in unique_years:
        year_df = predictions_df[predictions_df["Year"] == year]

        plt.figure(figsize=(8, 4))
        sns.histplot(year_df["AbsError"], bins=40, kde=False, color="steelblue")
        plt.title(f"Distribution of Absolute Errors: {year} → {year+1}", fontsize=14, fontweight="bold")
        plt.xlabel("Absolute Error")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.6)

        if save_plots:
            plt.savefig(f"{output_dir}/abs_error_hist_{year}.png", bbox_inches="tight")

        plt.show()


def plot_actual_vs_predicted_distributions(predictions_df, save_plots=False, output_dir="./evaluation_plots"):
    """
    Plot side-by-side histograms (overlaid) of actual vs predicted mortality rates per year.
    predictions_df must have columns: ["Year", "True", "Predicted"].
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)  # ✅ Create directory if it doesn't exist

    unique_years = sorted(predictions_df["Year"].unique())

    for year in unique_years:
        year_df = predictions_df[predictions_df["Year"] == year]

        plt.figure(figsize=(8, 4))
        # Actual distribution
        sns.histplot(year_df["True"], bins=40, color="steelblue", label="Actual", alpha=0.5, kde=False)
        # Predicted distribution
        sns.histplot(year_df["Predicted"], bins=40, color="orange", label="Predicted", alpha=0.5, kde=False)

        plt.title(f"Mortality Rate Distribution: {year} → {year+1}", fontsize=14, fontweight="bold")
        plt.xlabel("Mortality Rate")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        if save_plots:
            plt.savefig(f"{output_dir}/mortality_dist_comparison_squarederror_{year}.png", bbox_inches="tight")

        plt.show()



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


    ### Main script:
    ### Yearly prediction with k-fold CV within each year: 
    df = prepare_yearly_prediction_data()
    metrics, feature_importance, predictions, errors = yearly_mortality_prediction_native_categorical(df, n_splits=5)
    # top_2016 = get_top_feature_importance_for_year(feature_importance, year=2016, top_n=20)
    # plot_feature_importance(top_2016)
    #plot_grouped_feature_importance_all_years(feature_importance, show_plots=True)
    #plot_grouped_feature_importance_summary(feature_importance, start_year=2010, end_year=2021)
    # plot_feature_importance_all_years_native_categorical(feature_importance,
    #                                                     start_year=2010,
    #                                                     end_year=2021,
    #                                                     top_n=20,
    #                                                     save_dir="County Classification\XGB_Feat_Imp_Plots",
    #                                                     show_plots=True
    # )
    ### 8/29/25, EB: Added Andrew's big average feature importance plot function, using here:
    # plot_average_feature_importance(feature_importance)
    
    # print('Results from running XGBoost using Tweedie Objective function:')
    # print(metrics)
    # plot_abs_error_histograms(predictions, save_plots=False)
    plot_actual_vs_predicted_distributions(predictions, save_plots=True, output_dir='EB_Urbanicity/Plots/XGB_Mortality_Plots')
    


#######################################################################
### 8/18/25, EB: Hyperparameter tuning script, best results recorded below
# def main():    
    # df = prepare_yearly_prediction_data()
    # best_param_df = tune_xgb_hyperparameters_by_year(
    #                     df=df,
    #                     n_splits=5,
    #                     n_iter=30#,
    #                     #output_dir='xgb_tuning_results'
    #                 )
    # print("\nBest hyperparameters by year:")
    # print(best_param_df)



if __name__ == "__main__":
    main()


############################################################################
### 8/18/25, EB: Ran the grid search hyperparameter tuning function, tune_xgb_hyperparameters_by_year, and got the same hyperparameters for each year, thank god!
### Here's what I got:
### Best hyperparameters by year:
###     Year  Best RMSE  model__subsample  model__n_estimators  model__min_child_weight  model__max_depth  model__learning_rate  model__gamma  model__colsample_bytree
### 0   2010   8.565355               0.7                  500                        5                 7                  0.01             0                      0.8
### 1   2011   8.069581               0.7                  500                        5                 7                  0.01             0                      0.8
### 2   2012   8.188852               0.7                  500                        5                 7                  0.01             0                      0.8
### 3   2013   8.820456               0.7                  500                        5                 7                  0.01             0                      0.8
### 4   2014   9.329305               0.7                  500                        5                 7                  0.01             0                      0.8
### 5   2015  10.626819               0.7                  500                        5                 7                  0.01             0                      0.8
### 6   2016  11.503854               0.7                  500                        5                 7                  0.01             0                      0.8
### 7   2017  10.668884               0.7                  500                        5                 7                  0.01             0                      0.8
### 8   2018  11.156344               0.7                  500                        5                 7                  0.01             0                      0.8
### 9   2019  14.703657               0.7                  500                        5                 7                  0.01             0                      0.8
### 10  2020  17.634754               0.7                  500                        5                 7                  0.01             0                      0.8
### 11  2021  17.205477               0.7                  500                        5                 7                  0.01             0                      0.8