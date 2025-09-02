### 8/26/25, EB: Ok, so it seems like the simpler ordinal regression models from mord are pretty ill-suited to our problem. I could never get the scores
### at an even ok level, let alone good. However, there are ways to use custom loss functions in XGBoost, and I found a paper that describes several
### loss functions for use with ordinal regression tasks (https://home.ttic.edu/~nati/Publications/RennieSrebroIJCAI05.pdf)
### In this, the authors define the "all-threshold loss", where a greater penalty is incurred the farther away a prediction is from the correct bin.
### I think this makes a lot of sense for our current problem, and the nice thing about this, is that we can apply the loss to more than just logistic
### regression models. XGBoost is great for many reasons, and one is that it accepts custom loss functions, as long as you can define the first and sceond
### derivatives (gradient and hessian). This paper walks through that explicitly, so we can apply it directly!
### I'm thinking that if THIS still doesn't help, I'll need either outside direction from Adam or VM, or we switch to ranking algorithms. 

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import lognorm
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
 #from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, mean_absolute_error
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, kendalltau


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

def transform_to_categorical(df, svi_variables, bin_edges_dict=None, num_quantiles = 10):
    """
    ### 8/28/25, EB: Custom quantiles added
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
            ### 8/28/25, EB: Custom quantiles added
            # quantiles = np.linspace(0, 1, 10 + 1)  # 10 bins
            quantiles = np.linspace(0, 1, num_quantiles + 1)  # 10 bins
            edges = lognorm.ppf(quantiles, shape, loc=loc, scale=scale)
            bin_edges_dict[year] = edges

        edges = bin_edges_dict[year]
        pos_values = mort_year[is_positive]
        binned = np.digitize(pos_values, edges[1:], right=False)  # Class 1–9
        df.loc[year_mask & is_positive, 'mortality_bin'] = binned

    ### 8/26/25, EB: Just int, not pandas Int64, to prevent issues in the models farther down 
    # df['mortality_bin'] = df['mortality_bin'].astype('Int64')
    df['mortality_bin'] = df['mortality_bin'].astype('int')

    ## Keeping county_class as numerical for now
    # ----------------------
    # 3. Treat county class as categorical
    # ----------------------
    df['county_class'] = df['county_class'].astype('category')

    return df, bin_edges_dict


#############
### Custom XGBoost All-Threshold Ordinal Loss

# ---- All-threshold logistic loss ----
def make_all_threshold_logistic_obj(thresholds):
    """
    Returns an all-threshold logistic loss objective for XGBoost.
    
    thresholds : array-like of length (K-1)
        Bin boundaries between K ordinal classes.
        Must be increasing: θ1 < θ2 < ... < θ(K-1)
    """

    thresholds = np.array(thresholds)

    # def obj(y_true, y_pred):
    #     """
    #     y_true : DMatrix label array (ordinal ints in 0..K-1)
    #     y_pred : predicted scores (continuous, shape=(n,))
    #     """
    def obj(y_pred, dtrain):
        """
        y_pred : np.ndarray of shape (n,)
        dtrain : DMatrix
        """
        ### 8/26/25, EB: Got the error "margin = s * (y_pred - theta) TypeError: unsupported operand type(s) for -: 'DMatrix' and 'float'"
        ### Changed the y_true as follows:
        #y_true = y_true.astype(int)
        y_true = dtrain.get_label().astype(int)  # ✅ extract labels properly
        grad = np.zeros_like(y_pred)
        hess = np.zeros_like(y_pred)

        for l, theta in enumerate(thresholds, start=1):  # thresholds index from 1..K-1
            ### 8/27/25, EB: Flipped the sign from > to <=, seems to work much much better!
            #s = np.where(y_true > l, -1, 1)  # s(l;y): -1 if l < y else +1
            s = np.where(y_true <= l, -1, 1)  # s(l;y): -1 if l < y else +1
            margin = s * (y_pred - theta)
            #margin = s * (theta - y_pred)

            # logistic sigmoid
            p = 1.0 / (1.0 + np.exp(-margin))

            # gradient & hessian contributions
            grad += -s * (1 - p)
            hess += p * (1 - p)

        return grad, hess

    return obj

# ---- Prediction wrapper ----
def predict_classes(y_pred, thresholds):
    """
    Map raw scores -> ordinal bins using thresholds.
    """
    preds = np.digitize(y_pred, thresholds)  # returns bin index 0..K-1
    return preds


#############
### Model training/prediction loop

def yearly_mortality_prediction_allthreshold(df, bin_edges_dict, n_splits=5):
    """
    Predict next-year opioid mortality bins using XGBoost with
    custom all-threshold logistic loss and year-specific thresholds.
    Returns metrics, feature importances, and per-sample predictions.
    """

    metrics_all_years = []
    feature_importance_all = []
    all_predictions = []

    for year in range(2010, 2023):
        print(f"\n🔁 Processing year {year} → predicting mortality_bin for year {year+1}")
        df_year = df[df['year'] == year].copy()

        if df_year.empty or year not in bin_edges_dict:
            print(f"⚠️ Skipping year {year}: no data or missing thresholds.")
            continue

        feature_cols = [col for col in df.columns if col not in 
                        ['FIPS', 'year', 'mortality_rate', 'mortality_bin']]
        X = df_year[feature_cols].copy()
        y = df_year['mortality_bin'].astype(int)
        fips = df_year['FIPS']

        # Preserve categorical dtype for XGBoost
        if 'county_class' in X.columns:
            X['county_class'] = X['county_class'].astype('category')

        thresholds = bin_edges_dict[year]

        # Build custom loss for this year
        obj = make_all_threshold_logistic_obj(thresholds)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
            fips_test = fips.iloc[test_idx]

            # if 'county_class' in X_train.columns:
            #     X_train['county_class'] = X_train['county_class'].astype('category')
            #     X_test['county_class'] = X_test['county_class'].astype('category')

            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

            params = {
                "max_depth": 7,
                "eta": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "nthread": -1,
                "verbosity": 0
            }

            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=300,
                obj=obj
            )

            # Predict latent scores, then map to bins
            y_score = bst.predict(dtest)
            y_pred = predict_classes(y_score, thresholds)

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
            mae = mean_absolute_error(y_test, y_pred)
            fold_metrics.append({
                'fold': fold_idx + 1,
                'MAE': mae
            })

        # Average metrics across folds for this year
        fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
        year_metrics = fold_df.mean().to_dict()
        year_metrics['Year'] = year
        metrics_all_years.append(year_metrics)

    # Combine all results
    metrics_df = pd.DataFrame(metrics_all_years)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return metrics_df, predictions_df


#############
### Model performance evaluation

def evaluate_model_performance(metrics_df, predictions_df):
    """
    Summarize and visualize XGBoost ordinal model results.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Yearly summary metrics (MAE, Year, etc.).
    predictions_df : pd.DataFrame
        Per-sample predictions (FIPS, Year, True, Predicted, Fold).
    """

    # ---- Print metrics summary ----
    print("\n📊 Yearly Metrics:\n", metrics_df.round(4))
    avg_mae = metrics_df['MAE'].mean()
    print(f"\n✅ Overall Average MAE across years: {avg_mae:.4f}")

    # ---- Plot yearly MAE trend ----
    plt.figure(figsize=(8,5))
    sns.lineplot(data=metrics_df, x="Year", y="MAE", marker="o")
    plt.title("Yearly MAE Trend")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True, alpha=0.3)
    plt.show()

    # ---- Overall confusion matrix ----
    y_true = predictions_df["True"].astype(int)
    y_pred = predictions_df["Predicted"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=sorted(predictions_df["True"].unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=sorted(predictions_df["True"].unique()))
    fig, ax = plt.subplots(figsize=(7,7))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Overall Confusion Matrix (All Years)")
    plt.show()

    # ---- Per-year confusion matrices (optional) ----
    years = predictions_df["Year"].unique()
    for year in sorted(years):
        y_true_y = predictions_df.loc[predictions_df["Year"] == year, "True"].astype(int)
        y_pred_y = predictions_df.loc[predictions_df["Year"] == year, "Predicted"].astype(int)
        cm_y = confusion_matrix(y_true_y, y_pred_y, labels=sorted(predictions_df["True"].unique()))

        disp_y = ConfusionMatrixDisplay(confusion_matrix=cm_y,
                                        display_labels=sorted(predictions_df["True"].unique()))
        fig, ax = plt.subplots(figsize=(6,6))
        disp_y.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        plt.title(f"Confusion Matrix for Year {year}")
        plt.show()

### Adding ranking metrics: Spearman, Kendall's Tau, Top-K hit rate

def evaluate_model_performance_with_ranking(metrics_df, predictions_df, top_k=0.10):
    """
    Summarize and visualize XGBoost ordinal model results.
    Adds ranking-based evaluations (Spearman, Kendall, Top-K hit rate).
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Yearly summary metrics (MAE, Year, etc.).
    predictions_df : pd.DataFrame
        Per-sample predictions (FIPS, Year, True, Predicted, Fold).
    top_k : float
        Fraction for top-K evaluation (default=0.1 for top 10%).
    """

    # ---- Yearly MAE summary ----
    print("\n📊 Yearly Metrics:\n", metrics_df.round(4))
    avg_mae = metrics_df['MAE'].mean()
    print(f"\n✅ Overall Average MAE across years: {avg_mae:.4f}")

    # ---- Ranking metrics ----
    y_true = predictions_df["True"].astype(int)
    y_pred = predictions_df["Predicted"].astype(int)

    # Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)
    # Kendall's tau
    kendall_corr, _ = kendalltau(y_true, y_pred)

    print(f"\n📈 Spearman rank correlation: {spearman_corr:.4f}")
    print(f"📈 Kendall’s tau: {kendall_corr:.4f}")

    # ---- Top-K evaluation ----
    k = int(len(y_true) * top_k)
    # Indices of true top-k counties
    true_top_idx = y_true.sort_values(ascending=False).index[:k]
    pred_top_idx = y_pred.sort_values(ascending=False).index[:k]

    overlap = len(set(true_top_idx).intersection(set(pred_top_idx)))
    topk_hit_rate = overlap / k

    print(f"🎯 Top-{int(top_k*100)}% hit rate: {topk_hit_rate:.4f}")

    # ---- Return metrics for programmatic use ----
    results_summary = {
        "avg_mae": avg_mae,
        "spearman": spearman_corr,
        "kendall": kendall_corr,
        f"top_{int(top_k*100)}_hit_rate": topk_hit_rate
    }

    return results_summary


#############
### Model hyperparamter tuning loop
from itertools import product
from scipy.stats import spearmanr, kendalltau

def optimize_xgb_hyperparams(df, bin_edges_dict, 
                             param_grid=None, 
                             n_splits=5,
                             top_k=0.1):
    """
    Hyperparameter search for custom ordinal XGBoost model.
    
    Parameters
    ----------
    df : DataFrame
        Input dataset with features + mortality_bin.
    bin_edges_dict : dict
        Year-specific thresholds from log-normal fits.
    param_grid : dict
        Grid of hyperparams. Example:
        {
            "max_depth": [4, 6, 8],
            "eta": [0.05, 0.1],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "min_child_weight": [1, 5]
        }
    n_splits : int
        KFold splits per year.
    top_k : float
        Top-K fraction for hit rate evaluation.
    
    Returns
    -------
    best_params : dict
    results_table : pd.DataFrame
    """

    if param_grid is None:
        param_grid = {
            "max_depth": [5, 7],
            "eta": [0.05, 0.1],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "min_child_weight": [1, 5]
            # "num_boost_round": [300, 500]
        }

    # Expand grid
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in product(*values)]

    results = []

    for i, params in enumerate(combos, 1):
        print(f"\n🔎 Testing parameter set {i}/{len(combos)}: {params}")

        # Run model
        metrics_df, predictions_df = yearly_mortality_prediction_allthreshold_tuning(
            df, bin_edges_dict, n_splits=n_splits, params=params
        )

        # Aggregate metrics
        avg_mae = metrics_df["MAE"].mean()
        spearman_corr, _ = spearmanr(predictions_df["True"], predictions_df["Predicted"])
        kendall_corr, _ = kendalltau(predictions_df["True"], predictions_df["Predicted"])

        # Top-K hit rate
        k = int(len(predictions_df) * top_k)
        true_top = predictions_df["True"].sort_values(ascending=False).index[:k]
        pred_top = predictions_df["Predicted"].sort_values(ascending=False).index[:k]
        overlap = len(set(true_top).intersection(set(pred_top)))
        topk_hit_rate = overlap / k if k > 0 else np.nan

        results.append({
            **params,
            "avg_mae": avg_mae,
            "spearman": spearman_corr,
            "kendall": kendall_corr,
            f"top_{int(top_k*100)}%": topk_hit_rate
        })

    results_df = pd.DataFrame(results)

    # Choose best by lowest MAE (or you could rank by Spearman / Top-K)
    best_idx = results_df["avg_mae"].idxmin()
    best_params = results_df.loc[best_idx, param_grid.keys()].to_dict()

    print("\n🏆 Best parameters found:", best_params)
    print(results_df.sort_values("avg_mae"))

    return best_params, results_df

def yearly_mortality_prediction_allthreshold_tuning(df, bin_edges_dict, n_splits=5, params=None):
    """
    Predict next-year opioid mortality bins using XGBoost with
    custom all-threshold logistic loss and year-specific thresholds.
    
    Parameters
    ----------
    df : DataFrame
        Input data with features + mortality_bin.
    bin_edges_dict : dict
        Year-specific thresholds from log-normal fits.
    n_splits : int
        Number of folds for KFold CV.
    params : dict
        XGBoost hyperparameters (e.g. {"max_depth": 6, "eta": 0.05, ...}).
        If None, uses a default set.
    
    Returns
    -------
    metrics_df : DataFrame
        Yearly average metrics across folds.
    predictions_df : DataFrame
        Per-sample predictions (FIPS, Year, True, Predicted, Fold).
    """

    if params is None:
        params = {
            "max_depth": 7,
            "eta": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "nthread": -1,
            "verbosity": 0
        }

    metrics_all_years = []
    all_predictions = []

    for year in range(2010, 2023):
        print(f"\n🔁 Processing year {year} → predicting mortality_bin for year {year+1}")
        df_year = df[df['year'] == year].copy()

        if df_year.empty or year not in bin_edges_dict:
            print(f"⚠️ Skipping year {year}: no data or missing thresholds.")
            continue

        feature_cols = [col for col in df.columns if col not in 
                        ['FIPS', 'year', 'mortality_rate', 'mortality_bin']]
        X = df_year[feature_cols].copy()
        y = df_year['mortality_bin'].astype(int)
        fips = df_year['FIPS']

        if 'county_class' in X.columns:
            X['county_class'] = X['county_class'].astype('category')

        thresholds = bin_edges_dict[year]
        obj = make_all_threshold_logistic_obj(thresholds)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
            fips_test = fips.iloc[test_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

            bst = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=params.get("num_boost_round", 300),
                obj=obj
            )

            y_score = bst.predict(dtest)
            y_pred = predict_classes(y_score, thresholds)

            # Save predictions
            fold_df = pd.DataFrame({
                'FIPS': fips_test.values,
                'Year': year,
                'True': y_test.values,
                'Predicted': y_pred,
                'Fold': fold_idx + 1
            })
            all_predictions.append(fold_df)

            mae = mean_absolute_error(y_test, y_pred)
            fold_metrics.append({"fold": fold_idx + 1, "MAE": mae})

        fold_df = pd.DataFrame(fold_metrics).drop(columns='fold')
        year_metrics = fold_df.mean().to_dict()
        year_metrics['Year'] = year
        metrics_all_years.append(year_metrics)

    metrics_df = pd.DataFrame(metrics_all_years)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return metrics_df, predictions_df



#######################################################################################################################################################
def main():
    num_quantiles = 5
    print(f'Running Custom Ordinal XGBoost Regression Model with {100 / num_quantiles}% bins:')
    df = prepare_yearly_prediction_data()
    svi_variables = [v for v in DATA if v != 'Mortality']
    df, bin_edges_dict = transform_to_categorical(df, svi_variables, num_quantiles=num_quantiles)
    
    print('Data prepared, instantiating model:')
    print('-----------------------------------------')
    print('')
    metrics_df, predictions_df = yearly_mortality_prediction_allthreshold(df, bin_edges_dict, n_splits=5)
    evaluate_model_performance_with_ranking(metrics_df, predictions_df, top_k=0.05)
    print('Model prediction loop complete.')
    
    
# def main():
#     print('Hyperparameter Tuning for Custom Ordinal XGBoost Regression Model:')
#     df = prepare_yearly_prediction_data()
#     svi_variables = [v for v in DATA if v != 'Mortality']
#     df, bin_edges_dict = transform_to_categorical(df, svi_variables)
    
#     best_params, results_df = optimize_xgb_hyperparams(df, bin_edges_dict=bin_edges_dict, n_splits=5, top_k=0.10)
#     print('Model tuning complete.')
    

if __name__ == '__main__':
    main()

    
###############################################################
### Tuning results:
# 🏆 Best parameters found: {'max_depth': 7.0, 'eta': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.9, 'min_child_weight': 5.0}
#     max_depth   eta  subsample  colsample_bytree  min_child_weight   avg_mae  spearman   kendall   top_10%
# 19          7  0.05        0.7               0.9                 5  1.763279  0.678580  0.553996  0.422322
# 23          7  0.05        0.9               0.9                 5  1.765003  0.679059  0.554632  0.431601
# 22          7  0.05        0.9               0.9                 1  1.766088  0.680648  0.555619  0.429480
# 18          7  0.05        0.7               0.9                 1  1.766567  0.680076  0.555225  0.424973
# 17          7  0.05        0.7               0.7                 5  1.769030  0.678824  0.554026  0.428155
# 15          5  0.10        0.9               0.9                 5  1.771628  0.671544  0.546592  0.424443
# 21          7  0.05        0.9               0.7                 5  1.771683  0.679670  0.554665  0.429215
# 16          7  0.05        0.7               0.7                 1  1.771711  0.679168  0.554139  0.426034
# 3           5  0.05        0.7               0.9                 5  1.771787  0.681423  0.556872  0.435048
# 11          5  0.10        0.7               0.9                 5  1.772027  0.668868  0.543953  0.406416
# 2           5  0.05        0.7               0.9                 1  1.772109  0.680349  0.555769  0.431601
# 31          7  0.10        0.9               0.9                 5  1.772343  0.669142  0.544188  0.405620
# 7           5  0.05        0.9               0.9                 5  1.772582  0.681325  0.557033  0.432131
# 30          7  0.10        0.9               0.9                 1  1.773961  0.672045  0.547140  0.414104
# 10          5  0.10        0.7               0.9                 1  1.773986  0.668241  0.543289  0.395546
# 6           5  0.05        0.9               0.9                 1  1.775632  0.679467  0.555083  0.431866
# 14          5  0.10        0.9               0.9                 1  1.775922  0.669350  0.544746  0.415429
# 20          7  0.05        0.9               0.7                 1  1.775978  0.680484  0.555769  0.434252
# 26          7  0.10        0.7               0.9                 1  1.776132  0.668903  0.543891  0.402704
# 0           5  0.05        0.7               0.7                 1  1.776719  0.679866  0.555435  0.426564
# 1           5  0.05        0.7               0.7                 5  1.777274  0.680916  0.556469  0.424178
# 27          7  0.10        0.7               0.9                 5  1.777857  0.664847  0.539966  0.394486
# 8           5  0.10        0.7               0.7                 1  1.778043  0.667736  0.543133  0.410127
# 9           5  0.10        0.7               0.7                 5  1.778122  0.667799  0.543250  0.413839
# 13          5  0.10        0.9               0.7                 5  1.778547  0.669783  0.545003  0.419671
# 5           5  0.05        0.9               0.7                 5  1.779106  0.680747  0.556417  0.429480
# 4           5  0.05        0.9               0.7                 1  1.779317  0.680333  0.555918  0.429480
# 12          5  0.10        0.9               0.7                 1  1.779475  0.670352  0.545767  0.414104
# 25          7  0.10        0.7               0.7                 5  1.782260  0.664439  0.539772  0.399258
# 29          7  0.10        0.9               0.7                 5  1.783319  0.667379  0.542731  0.406151
# 28          7  0.10        0.9               0.7                 1  1.783771  0.670438  0.545895  0.408271
# 24          7  0.10        0.7               0.7                 1  1.787318  0.666724  0.542032  0.397667