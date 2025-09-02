### 8/19/25, EB: Alright, so the ordinal regression models didn't really work. I think there's just not enough information in
### the SVI variables to predict the ordinal categories well. The models did ok at predicting the low mortality counties, and slightly worse
### at predicting the high mortality counties, but everything else was just awful.
### I've thought about it, though, and in some sense I'm interested in the rank of the counties. Like, if I could rank the counties' mortality rates
### correctly, that would allow me to calculate the relative risk levels easily. So I'm going to try to use the XGBoost ranking model to predict the
### ranks of the counties' mortality rates.

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


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



def encode_features(df, svi_cols):
    """Encodes categorical variables (like county_class) and returns features + labels."""
    cat_col = ['county_class']
    num_cols = svi_cols

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), cat_col)
    ], remainder='passthrough')

    X = preprocessor.fit_transform(df[cat_col + num_cols])
    y = df['mortality_rate'].values
    return X, y, preprocessor

def train_xgb_ranker_sliding(data_df, svi_cols):
    from collections import defaultdict

    all_years = sorted(data_df['year'].unique())
    predictions_by_year = defaultdict(list)

    for year in all_years:
        train_df = data_df[data_df['year'] == year]
        test_df = data_df[data_df['year'] == year + 1] 

        if test_df.empty:
            continue

        # Encode and prepare data
        X_train, y_train, preproc = encode_features(train_df, svi_cols)
        X_test = preproc.transform(test_df[['county_class'] + svi_cols])
        y_test = test_df['mortality_rate'].values

        # Group: one group = one year of counties
        group_train = [len(train_df)]
        group_test = [len(test_df)]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(group_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        dtest.set_group(group_test)

        # XGBoost ranker config
        params = {
            "objective": "rank:pairwise",
            "eta": 0.1,
            "max_depth": 6,
            "eval_metric": "ndcg"
        }

        model = xgb.train(params, dtrain, num_boost_round=100)
        preds = model.predict(dtest)

        predictions_by_year[year + 1] = {
            'true': y_test,
            'pred': preds,
            'FIPS': test_df['FIPS'].values
        }

    return predictions_by_year

def train_xgb_ranker_expanding(data_df, svi_cols):
    from collections import defaultdict

    all_years = sorted(data_df['year'].unique())
    predictions_by_year = defaultdict(list)

    for i in range(1, len(all_years)):
        train_years = all_years[:i]       # Expanding window: years up to (but not including) the test year
        test_year = all_years[i]          # The "next" year is our test set

        train_df = data_df[data_df['year'].isin(train_years)]
        test_df = data_df[data_df['year'] == test_year]

        if test_df.empty or train_df.empty:
            continue

        # Encode and prepare data
        X_train, y_train, preproc = encode_features(train_df, svi_cols)
        X_test = preproc.transform(test_df[['county_class'] + svi_cols])
        y_test = test_df['mortality_rate'].values

        # Grouping (entire training set and entire test set each as one group)
        group_train = [len(train_df)]
        group_test = [len(test_df)]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(group_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        dtest.set_group(group_test)

        # XGBoost ranker config
        params = {
            "objective": "rank:pairwise",
            "eta": 0.1,
            "max_depth": 6,
            "eval_metric": "ndcg"
        }

        model = xgb.train(params, dtrain, num_boost_round=100)
        preds = model.predict(dtest)

        predictions_by_year[test_year] = {
            'true': y_test,
            'pred': preds,
            'FIPS': test_df['FIPS'].values
        }

    return predictions_by_year




def evaluate_ranking_performance(df, predictions_by_year, top_percent=10):
    """
    Evaluates ranking performance for each year, comparing predicted scores to actual top X% mortality rates.
    
    Args:
        df (pd.DataFrame): Your full prediction dataframe with columns 'FIPS', 'year', and 'mortality_rate'.
        predictions_by_year (dict): Dictionary of {year: predicted_scores_array}.
        top_percent (float): Percentile cutoff for defining high-risk counties (e.g., 10 for top 10%).

    Returns:
        pd.DataFrame: Per-year precision, recall, and F1 score.
    """
    results = []

    for year, preds in predictions_by_year.items():
        year_df = df[df['year'] == year].copy()
        ###### TROUBLESHOOTING ######
        # year_df['pred_score'] = preds
        year_df['pred_score'] = preds['pred']

        # 1. Define ground truth high-risk counties
        cutoff = year_df['mortality_rate'].quantile(1 - top_percent / 100.0)
        true_top = set(year_df[year_df['mortality_rate'] >= cutoff]['FIPS'])

        # 2. Define predicted top risk counties
        year_df_sorted = year_df.sort_values('pred_score', ascending=False)
        top_k = int(len(year_df_sorted) * top_percent / 100.0)
        pred_top = set(year_df_sorted.head(top_k)['FIPS'])

        # 3. Evaluation
        all_fips = list(year_df['FIPS'])
        y_true = [1 if fips in true_top else 0 for fips in all_fips]
        y_pred = [1 if fips in pred_top else 0 for fips in all_fips]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            'year': year,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    return pd.DataFrame(results)

def plot_ranking_performance(results_df, top_percent=10):
    """
    Plots precision, recall, and F1 score over the years.
    
    Args:
        results_df (pd.DataFrame): DataFrame with columns ['year', 'precision', 'recall', 'f1'].
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['year'], results_df['precision'], marker='o', label='Precision')
    plt.plot(results_df['year'], results_df['recall'], marker='^', label='Recall')
    plt.plot(results_df['year'], results_df['f1'], marker='s', label='F1 Score')


    plt.title("Top {}% Risk Prediction Performance".format(top_percent))
    plt.xlabel('Year')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()

# def main():
#     print("First, sliding window approach:")
#     # Prepare data
#     df = prepare_yearly_prediction_data()
#     svi_cols = [col for col in DATA if col != 'Mortality']

#     # Train XGB ranker
#     predictions_by_year = train_xgb_ranker_sliding(df, svi_cols)

#     # Evaluate performance
#     results_df = evaluate_ranking_performance(df, predictions_by_year, top_percent=10)

#     # Plot results
#     plot_ranking_performance(results_df, top_percent=10)

#     print("Now, expanding window approach:")
#     # Expanding window approach
#     expanding_preds = train_xgb_ranker_expanding(df, svi_cols)
#     expanding_results = evaluate_ranking_performance(df, expanding_preds, top_percent=10)
#     plot_ranking_performance(expanding_results, top_percent=10)

    
def main():
    ### This one evaluates the perofmrance of the two models on different top percentiles of risk counties.
    # Prepare data
    df = prepare_yearly_prediction_data()
    svi_cols = [col for col in DATA if col != 'Mortality']

    # Train XGB ranker
    sliding_preds = train_xgb_ranker_sliding(df, svi_cols)
    # Expanding window approach
    expanding_preds = train_xgb_ranker_expanding(df, svi_cols)


    top_thresholds = [1, 2, 5, 10, 15, 20, 25]
    for top_percent in top_thresholds:
        print(f"Evaluating top {top_percent}% risk counties...")

        # Evaluate sliding window performance
        sliding_results = evaluate_ranking_performance(df, sliding_preds, top_percent=top_percent)
        # Plot results
        plot_ranking_performance(sliding_results, top_percent=top_percent)

        # Evaluate expanding window performance
        expanding_results = evaluate_ranking_performance(df, expanding_preds, top_percent=top_percent)
        plot_ranking_performance(expanding_results, top_percent=top_percent)
    


if __name__ == "__main__":
    main()