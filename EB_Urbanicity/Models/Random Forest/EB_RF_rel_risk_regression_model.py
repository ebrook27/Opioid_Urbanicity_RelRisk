### 4/3/25, EB: Adam said don't try to predict the county labels, rather we should try to include them as input using as one-hot encoding. 
### Then we can predict mortality rates. Andrew suggested a random forest regression model, which I think is a good place to start.

###########################################################################################################
### 4/3/25, EB: Refactored the code to fit a new prepare_temporal_data function.

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# DATA = ['Mortality',
#         'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
#         'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
#         'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
#         'Single-Parent Household', 'Unemployment']

# def prepare_temporal_data():
#     """Loads and reshapes data for regression, including mortality and county class."""
#     all_data = []
#     svi_variables = [v for v in DATA if v != 'Mortality']
    
#     for variable in svi_variables:
#         var_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
#         var_df = pd.read_csv(var_path, dtype={'FIPS': str})
#         var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
#         yearly_cols = [f'{year} {variable}' for year in range(2010, 2023)]
#         var_data = var_df[['FIPS'] + yearly_cols].set_index('FIPS')
#         all_data.append(var_data)
    
#     # Combine all SVI variables (per year)
#     combined_df = pd.concat(all_data, axis=1)
#     #print(combined_df.head())

#     # Load mortality data (target variable)
#     mortality_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
#     mortality_df['FIPS'] = mortality_df['FIPS'].str.zfill(5)
#     # Use a single year or average over years
#     mortality_df['mortality_rate'] = mortality_df[[f'{year} MR' for year in range(2010, 2023)]].mean(axis=1)
#     mortality_df = mortality_df[['FIPS', 'mortality_rate']].set_index('FIPS')

#     # Load county category (0–5) as categorical variable
#     nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
#     nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
#     nchs_df = nchs_df.set_index('FIPS')
#     nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)  # Treat as categorical
#     nchs_df = nchs_df[['county_class']]

#     # Merge everything
#     final_df = combined_df.join([mortality_df, nchs_df], how='inner')
#     final_df = final_df.dropna()

#     # Reshape features: flatten (13 years * N SVI vars)
#     n_years = 13
#     n_features = len(svi_variables)
#     X = final_df.drop(columns=['mortality_rate', 'county_class']).values.reshape(-1, n_years * n_features)

#     # Return X, y, and county class for later one-hot encoding
#     y = final_df['mortality_rate'].values
#     county_class = final_df['county_class'].values

#     return X, y, county_class, final_df.index  # Also return FIPS index if needed

# # Get data
# X, y, county_class, fips = prepare_temporal_data()
# # print(X[:5])
# # print(y[:5])
# # print(county_class[:5])
# # print(fips[:5])



# # Combine X and county class into a DataFrame
# X_df = pd.DataFrame(X)
# X_df['county_class'] = county_class
# # print("X_df.head:", X_df.head())
# # print("X_df Columns:", X_df.columns)
# #print("X_df shape:", X_df.shape)

# # Fix mixed column types:
# X_df.columns = X_df.columns.astype(str)

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# # Build preprocessing + regression pipeline
# preprocessor = ColumnTransformer(
#     transformers=[('county_class', OneHotEncoder(drop='first'), ['county_class'])],
#     remainder='passthrough'
# )

# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', DecisionTreeRegressor(max_depth=5, random_state=42))
# ])

# pipeline.fit(X_train, y_train)

# # Predict on full dataset to rank risk
# y_pred = pipeline.predict(X_df)
# results_df = X_df.copy()
# results_df['predicted_risk'] = y_pred
# #results_df['actual_cases'] = y > 0  # or some threshold if needed
# results_df['mortality_rate'] = y
# threshold = results_df['mortality_rate'].quantile(0.9)
# results_df['actual_cases'] = results_df['mortality_rate'] > threshold


# print(results_df['predicted_risk'].describe())
# print(results_df['actual_cases'].value_counts(normalize=True))


# plt.hist(results_df['predicted_risk'], bins=30)
# plt.title("Distribution of Predicted Risk")
# plt.xlabel("Predicted Risk")
# plt.ylabel("Count")
# plt.show()




# def compute_relative_risk(df, strata_percentages, prediction_col='predicted_risk', label_col='actual_cases'):
#     total_cases = df[label_col].sum()
#     total_samples = len(df)
    
#     results = []
#     for p in strata_percentages:
#         top_n = int(total_samples * p)
#         top_df = df.nlargest(top_n, prediction_col)
#         top_cases = top_df[label_col].sum()
#         top_samples = len(top_df)
#         rr = (top_cases / top_samples) / (total_cases / total_samples)
#         results.append({'Stratum': f'Top {p*100:.1f}%', 'Relative Risk': rr})
    
#     return pd.DataFrame(results)

# rr_df = compute_relative_risk(results_df, [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.2])
# print(rr_df)


# # Plotting relative risk by risk stratum
# def plot_relative_risk(rr_df):
#     plt.figure(figsize=(8, 5))
#     sns.barplot(x='Stratum', y='Relative Risk', data=rr_df)

#     # Reference line at RR = 1
#     plt.axhline(1, color='gray', linestyle='--', label='Baseline Risk (RR=1)')

#     # Add text labels on bars
#     for index, row in rr_df.iterrows():
#         plt.text(index, row['Relative Risk'] + 0.05, f"{row['Relative Risk']:.2f}", 
#                  ha='center', va='bottom', fontsize=10)

#     plt.title("Relative Risk by Predicted Risk Stratum")
#     plt.ylabel("Relative Risk")
#     plt.xlabel("Predicted Risk Stratum")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # Call it
# plot_relative_risk(rr_df)



################################################################################################################################################
################################################################################################################################################
### 4/4/25, EB: The above code does an aggregated prediction across all years. What I'd like to investigate now
### is a yearly prediction. By this I mean taking the data for a single year and predicting the mortality rate for the following year.
### This would be a more realistic prediction of mortality risk, as it would be based on the most recent data available.
### I will create a new function to prepare the data for this yearly prediction.

### 4/7/25, EB: I am adding a couple more metrics to evaluate the model performance. Since we are interested in predicting the ranking of the counties, 
### I will add a top-k recall metric. I will also add a normalized discounted cumulative gain (NDCG) metric, which is a common metric for evaluating ranking models.

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
import numpy as np


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

## The following functions are used in the yearly_prediction function to compute relative risk and top-k recall metrics.
def compute_relative_risk(df, strata_percentages, prediction_col='predicted_risk', label_col='actual_cases'):
    total_cases = df[label_col].sum()
    total_samples = len(df)

    results = []
    for p in strata_percentages:
        top_n = max(1, int(total_samples * p))  # At least one
        top_df = df.nlargest(top_n, prediction_col)
        top_cases = top_df[label_col].sum()
        top_samples = len(top_df)
        rr = (top_cases / top_samples) / (total_cases / total_samples)
        results.append({'Stratum': f'Top {p*100:.1f}%', 'Relative Risk': rr})

    return pd.DataFrame(results)

def top_k_recall(y_true, y_pred, k_fraction=0.05):
    """
    Compute Top-K Recall: how many of the top actual mortality counties are captured
    in the top-k predicted risk counties.

    Parameters:
        y_true: Ground truth mortality rates
        y_pred: Predicted mortality rates
        k_fraction: Fraction of samples to consider (e.g., 0.05 for top 5%)

    Returns:
        recall_score: proportion of top-actual counties captured in top-predicted
    """
    n = len(y_true)
    k = max(1, int(n * k_fraction))

    top_actual_idx = set(y_true.argsort()[-k:][::-1])
    top_pred_idx = set(y_pred.argsort()[-k:][::-1])

    intersection = len(top_actual_idx & top_pred_idx)
    recall_score = intersection / k
    return recall_score

def compute_ndcg(y_true, y_pred, k_fraction=0.05):
    n = len(y_true)
    k = max(1, int(n * k_fraction))

    y_true_array = np.asarray(y_true).reshape(1, -1)
    y_pred_array = np.asarray(y_pred).reshape(1, -1)

    return ndcg_score(y_true_array, y_pred_array, k=k)


# Loop through each year: train on y, predict for y+1
## This loop computed mortality rates for the same year as the data, which is not what we want.
## We want to predict the mortality rate for the next year, so we need to train on the current year and predict for the next year.
# for year in range(2010, 2022):
#     print(f"🔁 Processing year {year} → predicting {year+1}")

#     # Split data
#     train_df = df[df['year'] == year].copy()
#     test_df = df[df['year'] == year].copy()  # We'll predict for this year, and evaluate on true mortality in y+1

#     # Drop rows with missing target
#     test_df = test_df.dropna(subset=['mortality_rate'])

#     # Features and target
#     feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_rate']]
#     X_train = train_df[feature_cols]
#     y_train = train_df['mortality_rate']
#     X_test = test_df[feature_cols]
#     y_test = test_df['mortality_rate']

#     # Pipeline: One-hot encode county_class
#     preprocessor = ColumnTransformer([
#         ('cat', OneHotEncoder(drop='first'), ['county_class'])
#     ], remainder='passthrough')

#     ### 4/4/25, EB: Switched to RandomForestRegressor for see if there's any  performance gains to be had.

#     pipeline = Pipeline([
#     ('prep', preprocessor),
#     ('model', RandomForestRegressor(
#         n_estimators=100,
#         max_depth=10,
#         random_state=42,
#         n_jobs=-1
#         ))
#     ])
#     # pipeline = Pipeline([
#     #     ('prep', preprocessor),
#     #     ('model', DecisionTreeRegressor(max_depth=5, random_state=42))
#     # ])

#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
    
#     # Prediction accuracy
#     rmse = mean_squared_error(y_test, y_pred, squared=False)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     metrics_all_years.append({'Year': year, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

#     # Construct results for evaluation
#     results = test_df.copy()
#     results['predicted_risk'] = y_pred

#     # Define extreme cases (top 10% of mortality in actual data)
#     threshold = results['mortality_rate'].quantile(0.8)
#     results['actual_cases'] = results['mortality_rate'] > threshold

#     # Compute relative risk
#     rr = compute_relative_risk(results, [0.001, 0.01, 0.05, 0.10, 0.15, 0.2])
#     rr['Year'] = year
#     rr_all_years.append(rr)

def yearly_prediction(df):
    """Predicts mortality rates for the next year using current-year SVI + county class."""
    ### 4/8/25, EB: Realized I was constructing the y values (mortality) incorrectly. In our constructor
    ### above, we already have the mortality rate for the next year, so we can just use that directly.
    ### As it was, we were using the mortality rate for two years ahead, not just one. 
    
    rr_all_years = []
    metrics_all_years = []
    
    # Loop through each year: train on y, predict for y+1
    for year in range(2010, 2021):
        # print(f"🔁 Processing year {year} → predicting {year+1}")

        # # Get features from year n
        # X_year_n = df[df['year'] == year].copy()
        # print(X_year_n.head())
        
        # # Get mortality from year n+1
        # y_year_n1 = df[df['year'] == year + 1][['FIPS', 'mortality_rate']].copy()
        # print(y_year_n1.head())

        # # Merge features + target
        # merged = X_year_n.merge(y_year_n1, on='FIPS', how='inner', suffixes=('', '_target'))

        # if merged.empty:
        #     print(f"⚠️ Skipping year {year} → {year+1}: no overlapping counties.")
        #     continue

        # feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_rate']]
        # X = merged[feature_cols]
        # y = merged['mortality_rate_target']
        
        print(f"🔁 Processing year {year} → predicting mortality in year {year+1}")

        df_year = df[df['year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            continue

        feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_rate']]
        X = df_year[feature_cols]
        y = df_year['mortality_rate']  # already aligned with year+1 mortality

        # 💡 Split counties into train/test groups
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocessing + model
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), ['county_class'])
        ], remainder='passthrough')

        pipeline = Pipeline([
            ('prep', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=250,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Accuracy on true held-out counties
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        
        # Ranking metrics
        top1 = top_k_recall(y_test.values, y_pred, k_fraction=0.01)
        top5 = top_k_recall(y_test.values, y_pred, k_fraction=0.05)
        ndcg = compute_ndcg(y_test.values, y_pred, k_fraction=0.10)

        #metrics_all_years.append({'Year': year, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        metrics_all_years.append({
        'Year': year,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Top1%_Recall': top1,
        'Top5%_Recall': top5,
        'nDCG@10%': ndcg
        })


        # Evaluate relative risk on test set
        results = X_test.copy()
        results['predicted_risk'] = y_pred
        results['mortality_rate'] = y_test.values

        #threshold = results['mortality_rate'].quantile(0.9)
        results['actual_cases'] = results['mortality_rate'] > 0#threshold
        #print(results['actual_cases'].value_counts())

        rr = compute_relative_risk(results, [0.001, 0.01, 0.05, 0.10])
        rr['Year'] = year
        rr_all_years.append(rr)
        
    # Combine all RR results
    rr_df = pd.concat(rr_all_years)
    metrics_df = pd.DataFrame(metrics_all_years)
    
    return rr_df, metrics_df

def plot_results(rr_df, metrics_df):

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rr_df, x='Year', y='Relative Risk', hue='Stratum', marker='o')
    plt.title("Relative Risk of Overdose Mortality Across Years by Stratum")
    plt.ylabel("Relative Risk")
    plt.xlabel("Prediction Year")
    plt.axhline(1, linestyle='--', color='gray')
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10, 6))
    sns.lineplot(data=metrics_df, x='Year', y='RMSE', marker='o', label='RMSE')
    sns.lineplot(data=metrics_df, x='Year', y='MAE', marker='o', label='MAE')
    plt.title("Prediction Error Over Time")
    plt.ylabel("Error")
    plt.xlabel("Year")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=metrics_df, x='Year', y='R2', marker='o')
    plt.axhline(0, linestyle='--', color='gray')
    plt.title("R² Score Over Time")
    plt.ylabel("R²")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10, 6))
    sns.lineplot(data=metrics_df, x='Year', y='Top1%_Recall', label='Top 1% Recall')
    sns.lineplot(data=metrics_df, x='Year', y='Top5%_Recall', label='Top 5% Recall')
    plt.title("Top-K Recall over Time")
    plt.ylabel("Recall")
    plt.xlabel("Prediction Year")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=metrics_df, x='Year', y='nDCG@10%', marker='o')
    plt.title("nDCG@10% over Time")
    plt.ylabel("nDCG")
    plt.xlabel("Prediction Year")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def main():
    df = prepare_yearly_prediction_data()

    rr_df, metrics_df = yearly_prediction(df)
    # print(rr_df.head())
    # print(metrics_df.head())
    # print(metrics_df.describe())

    # Plot results
    
    plot_results(rr_df, metrics_df)


if __name__ == "__main__":
    main()