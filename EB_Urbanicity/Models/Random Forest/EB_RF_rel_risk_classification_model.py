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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
from collections import Counter
warnings.filterwarnings("ignore", message=".*use_inf_as_na option is deprecated.*")


custom_levels_path = 'Data\Mortality\Final Files\Mortality_relative_risk_custom_levels.csv'
even_levels_path = 'Data\Mortality\Final Files\Mortality_relative_risk_levels.csv'

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

def prepare_yearly_prediction_data(mortality_path):
    """Creates a long-format dataset for predicting next-year mortality using current-year SVI + county class."""
    svi_variables = [v for v in DATA if v != 'Mortality']
    years = list(range(2010, 2022))  # We predict mortality up to 2022

    # Load county category (static)
    nchs_df = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    nchs_df['FIPS'] = nchs_df['FIPS'].str.zfill(5)
    nchs_df = nchs_df.set_index('FIPS')
    nchs_df['county_class'] = nchs_df['2023 Code'].astype(str)
    
    # Load mortality
    #mort_df = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
    # rr_df = pd.read_csv('Data\Mortality\Final Files\Mortality_relative_risk_custom_levels.csv', dtype={'FIPS': str})
    rr_df = pd.read_csv(mortality_path, dtype={'FIPS': str})
    rr_df['FIPS'] = rr_df['FIPS'].str.zfill(5)
    rr_df = rr_df.set_index('FIPS')
    
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

    # 🔁 Add binned relative risk level for year+1
    for y in years:
        rr_col = f'{y+1} RR_Level'
        if rr_col not in rr_df.columns:
            continue
        svi_merged.loc[svi_merged['year'] == y, 'rr_bin'] = svi_merged.loc[svi_merged['year'] == y, 'FIPS'].map(rr_df[rr_col])
    
    svi_merged['rr_bin'] = svi_merged['rr_bin'].astype(int)

    # Add county class
    svi_merged = svi_merged.merge(nchs_df[['county_class']], on='FIPS', how='left')

    # Drop rows with any missing values
    svi_merged = svi_merged.dropna()

    return svi_merged

def yearly_classification_prediction(df, n_bins=20, n_splits=5):
    """
    Classifies counties into binned relative risk classes (0–n_bins) for each year.
    Uses cross-validation within each year for robustness.
    """
    metrics_all_years = []

    
#    with open(logfile_path, "w") as log:
    for year in range(2010, 2021):
        print(f"🔁 Processing year {year} → predicting RR bin for year {year+1}")
        # log.write(f"- Processing year {year} -> predicting RR bin for year {year+1}\n")
        df_year = df[df['year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ Skipping year {year}: no data.")
            # log.write(f"!! Skipping year {year}: no data.\n")
            continue

        # Drop counties with RR=0 or very low values if needed
        # df_year = df_year[df_year['relative_risk_score'] > 0]

        feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'relative_risk_score']]
        X = df_year[feature_cols]
        #y_rr = df_year['relative_risk_score']
        y_class = df_year['rr_bin']  # Already an integer class label

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_class)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]

            ## 4/10/25, EB: Wanted to make sure that the folds were balanced in terms of class distribution. Seems like they are!
            # train_dist = Counter(y_class.iloc[train_idx])
            # test_dist = Counter(y_class.iloc[test_idx])
            # print(f"\nFold {fold_idx+1}")
            # print("Train class distribution:", train_dist)
            # print("Test class distribution:", test_dist)
            
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
            ### 4/8/25, EB:  Debugging, I think we might not be breaking the data into folds evenly.
            #print(f"Fold {fold_idx+1} — Unique predicted classes: {np.unique(y_pred)}")
            # print(f"📄 Classification report for year {year}, fold {fold_idx+1}")
            # print(classification_report(y_test, y_pred, digits=3))
            # log.write(f"\n Classification report for year {year}, fold {fold_idx+1}\n")
            # log.write(classification_report(y_test, y_pred, digits=3))

            # conf_mat = confusion_matrix(y_test, y_pred, labels=range(n_bins))
            # # print(f"🔀 Confusion matrix (year {year}, fold {fold_idx+1}):")
            # # print(conf_mat)
            # log.write(f"\n Confusion matrix (year {year}, fold {fold_idx+1}):\n")
            # log.write(np.array2string(conf_mat, separator=', ') + "\n")
            
            # unique, counts = np.unique(y_pred, return_counts=True)
            # pred_dist = dict(zip(unique, counts))
            # # print("🧮 Predicted class distribution:", dict(zip(unique, counts)))
            # log.write(f" Predicted class distribution:\n{pred_dist}\n\n")
            
            # acc = accuracy_score(y_test, y_pred)
            # f1 = f1_score(y_test, y_pred, average='macro')
            # recall = recall_score(y_test, y_pred, average='macro')
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)


            fold_metrics.append({'fold': fold_idx+1, 'Accuracy': acc, 'F1_macro': f1, 'Recall_macro': recall})

        # Aggregate metrics for the year
        #print(fold_metrics)
        fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
        year_metrics = fold_df.mean().to_dict()
        year_metrics['Year'] = year
        metrics_all_years.append(year_metrics)

    metrics_df = pd.DataFrame(metrics_all_years)
    return metrics_df


# def plot_results(rr_df, metrics_df):

#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=rr_df, x='Year', y='Relative Risk', hue='Stratum', marker='o')
#     plt.title("Relative Risk of Overdose Mortality Across Years by Stratum")
#     plt.ylabel("Relative Risk")
#     plt.xlabel("Prediction Year")
#     plt.axhline(1, linestyle='--', color='gray')
#     plt.tight_layout()
#     plt.show()


#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=metrics_df, x='Year', y='RMSE', marker='o', label='RMSE')
#     sns.lineplot(data=metrics_df, x='Year', y='MAE', marker='o', label='MAE')
#     plt.title("Prediction Error Over Time")
#     plt.ylabel("Error")
#     plt.xlabel("Year")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(10, 5))
#     sns.lineplot(data=metrics_df, x='Year', y='R2', marker='o')
#     plt.axhline(0, linestyle='--', color='gray')
#     plt.title("R² Score Over Time")
#     plt.ylabel("R²")
#     plt.xlabel("Year")
#     plt.tight_layout()
#     plt.show()


#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=metrics_df, x='Year', y='Top1%_Recall', label='Top 1% Recall')
#     sns.lineplot(data=metrics_df, x='Year', y='Top5%_Recall', label='Top 5% Recall')
#     plt.title("Top-K Recall over Time")
#     plt.ylabel("Recall")
#     plt.xlabel("Prediction Year")
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(10, 5))
#     sns.lineplot(data=metrics_df, x='Year', y='nDCG@10%', marker='o')
#     plt.title("nDCG@10% over Time")
#     plt.ylabel("nDCG")
#     plt.xlabel("Prediction Year")
#     plt.ylim(0, 1)
#     plt.tight_layout()
#     plt.show()


def plot_results(metrics_df):
    """
    Plot accuracy, F1 (macro), and recall (macro) over prediction years.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=metrics_df, x='Year', y='Accuracy', label='Accuracy', marker='o')
    sns.lineplot(data=metrics_df, x='Year', y='F1_macro', label='F1 Score (Macro)', marker='s')
    sns.lineplot(data=metrics_df, x='Year', y='Recall_macro', label='Recall (Macro)', marker='^')

    plt.title("Model Classification Performance Over Time")
    plt.ylabel("Score")
    plt.xlabel("Prediction Year")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def main():
    #logfile_path = "classification_debug_log.txt"
    df = prepare_yearly_prediction_data(custom_levels_path)
    #print(df.head())
    
    metrics_df = yearly_classification_prediction(df)#, logfile_path=logfile_path)
    # print(rr_df.head())
    # print(metrics_df.head())
    # print(metrics_df.describe())

    # Plot results
    plot_results(metrics_df)
    #print(metrics_df)

if __name__ == "__main__":
    main()






















## The following functions are used in the yearly_prediction function to compute relative risk and top-k recall metrics.
# def compute_relative_risk(df, strata_percentages, prediction_col='predicted_risk', label_col='actual_cases'):
#     total_cases = df[label_col].sum()
#     total_samples = len(df)

#     results = []
#     for p in strata_percentages:
#         top_n = max(1, int(total_samples * p))  # At least one
#         top_df = df.nlargest(top_n, prediction_col)
#         top_cases = top_df[label_col].sum()
#         top_samples = len(top_df)
#         rr = (top_cases / top_samples) / (total_cases / total_samples)
#         results.append({'Stratum': f'Top {p*100:.1f}%', 'Relative Risk': rr})

#     return pd.DataFrame(results)

# def top_k_recall(y_true, y_pred, k_fraction=0.05):
#     """
#     Compute Top-K Recall: how many of the top actual mortality counties are captured
#     in the top-k predicted risk counties.

#     Parameters:
#         y_true: Ground truth mortality rates
#         y_pred: Predicted mortality rates
#         k_fraction: Fraction of samples to consider (e.g., 0.05 for top 5%)

#     Returns:
#         recall_score: proportion of top-actual counties captured in top-predicted
#     """
#     n = len(y_true)
#     k = max(1, int(n * k_fraction))

#     top_actual_idx = set(y_true.argsort()[-k:][::-1])
#     top_pred_idx = set(y_pred.argsort()[-k:][::-1])

#     intersection = len(top_actual_idx & top_pred_idx)
#     recall_score = intersection / k
#     return recall_score

# def compute_ndcg(y_true, y_pred, k_fraction=0.05):
#     n = len(y_true)
#     k = max(1, int(n * k_fraction))

#     y_true_array = np.asarray(y_true).reshape(1, -1)
#     y_pred_array = np.asarray(y_pred).reshape(1, -1)

#     return ndcg_score(y_true_array, y_pred_array, k=k)


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

# def yearly_prediction(df):
#     """Predicts mortality rates for the next year using current-year SVI + county class."""
#     ### 4/8/25, EB: Realized I was constructing the y values (mortality) incorrectly. In our constructor
#     ### above, we already have the mortality rate for the next year, so we can just use that directly.
#     ### As it was, we were using the mortality rate for two years ahead, not just one. 
    
#     rr_all_years = []
#     metrics_all_years = []
    
#     # Loop through each year: train on y, predict for y+1
#     for year in range(2010, 2021):
#         # print(f"🔁 Processing year {year} → predicting {year+1}")

#         # # Get features from year n
#         # X_year_n = df[df['year'] == year].copy()
#         # print(X_year_n.head())
        
#         # # Get mortality from year n+1
#         # y_year_n1 = df[df['year'] == year + 1][['FIPS', 'mortality_rate']].copy()
#         # print(y_year_n1.head())

#         # # Merge features + target
#         # merged = X_year_n.merge(y_year_n1, on='FIPS', how='inner', suffixes=('', '_target'))

#         # if merged.empty:
#         #     print(f"⚠️ Skipping year {year} → {year+1}: no overlapping counties.")
#         #     continue

#         # feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_rate']]
#         # X = merged[feature_cols]
#         # y = merged['mortality_rate_target']
        
#         print(f"🔁 Processing year {year} → predicting mortality in year {year+1}")

#         df_year = df[df['year'] == year].copy()

#         if df_year.empty:
#             print(f"⚠️ Skipping year {year}: no data.")
#             continue

#         feature_cols = [col for col in df.columns if col not in ['FIPS', 'year', 'mortality_rate']]
#         X = df_year[feature_cols]
#         y = df_year['mortality_rate']  # already aligned with year+1 mortality

#         # 💡 Split counties into train/test groups
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         # Preprocessing + model
#         preprocessor = ColumnTransformer([
#             ('cat', OneHotEncoder(drop='first'), ['county_class'])
#         ], remainder='passthrough')

#         pipeline = Pipeline([
#             ('prep', preprocessor),
#             ('model', RandomForestRegressor(
#                 n_estimators=250,
#                 max_depth=15,
#                 random_state=42,
#                 n_jobs=-1
#             ))
#         ])

#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_test)

#         # Accuracy on true held-out counties
#         rmse = mean_squared_error(y_test, y_pred, squared=False)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
        
        
#         # Ranking metrics
#         top1 = top_k_recall(y_test.values, y_pred, k_fraction=0.01)
#         top5 = top_k_recall(y_test.values, y_pred, k_fraction=0.05)
#         ndcg = compute_ndcg(y_test.values, y_pred, k_fraction=0.10)

#         #metrics_all_years.append({'Year': year, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
#         metrics_all_years.append({
#         'Year': year,
#         'RMSE': rmse,
#         'MAE': mae,
#         'R2': r2,
#         'Top1%_Recall': top1,
#         'Top5%_Recall': top5,
#         'nDCG@10%': ndcg
#         })


#         # Evaluate relative risk on test set
#         results = X_test.copy()
#         results['predicted_risk'] = y_pred
#         results['mortality_rate'] = y_test.values

#         #threshold = results['mortality_rate'].quantile(0.9)
#         results['actual_cases'] = results['mortality_rate'] > 0#threshold
#         #print(results['actual_cases'].value_counts())

#         rr = compute_relative_risk(results, [0.001, 0.01, 0.05, 0.10])
#         rr['Year'] = year
#         rr_all_years.append(rr)
        
#     # Combine all RR results
#     rr_df = pd.concat(rr_all_years)
#     metrics_df = pd.DataFrame(metrics_all_years)
    
#     return rr_df, metrics_df

