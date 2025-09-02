### 8/22/25, EB: Ok, I think making the problem purely categorical might be helpful, but even when I was using a tuned XGBoost classifier, the results were still pretty bad.
### This is probably at least partially due to flattening the ordinal nature of the problem, because XGBoostClassifier is not designed for ordinal classification.
### I found a paper that uses a simple method to turn a multi-class classifier into an ordinal regression model, and it seems like it would be useful and easy to implement.
### The method involves performing several binary classifications, one for each cut point between the bins, and then combining the results to get the final probability distribution over the bins.
### There's an implementation of this in the `xgboostordinal` package, so I'm going to try that next.
### I should be able to use the same data preparation and feature engineering steps, so I won't need to change much of the code.



import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
from tqdm import tqdm
from itertools import product
from xgbordinal import XGBOrdinal

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


def transform_to_categorical(df, svi_variables, bin_edges_dict=None):
    """
    Transforms the DataFrame into categorical format:
    - SVI variables binned into deciles (0–9)
    - Mortality rate binned into: 0 (no mortality) + 9 bins (log-normal)
    - County class cast as categorical

    Optionally accepts precomputed bin_edges_dict for mortality bins.
    Returns transformed DataFrame + bin_edges_dict for reuse.
    """
    df = df.copy()
    
    # ----------------------
    # 1. Bin SVI variables into deciles
    # ----------------------
    for var in svi_variables:
        df[f'{var}_bin'] = pd.qcut(df[var], 10, labels=False, duplicates='drop')

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
            quantiles = np.linspace(0, 1, 10 + 1)  # 10 bins
            edges = lognorm.ppf(quantiles, shape, loc=loc, scale=scale)
            bin_edges_dict[year] = edges

        edges = bin_edges_dict[year]
        pos_values = mort_year[is_positive]
        binned = np.digitize(pos_values, edges[1:], right=False)  # Class 1–9
        df.loc[year_mask & is_positive, 'mortality_bin'] = binned

    df['mortality_bin'] = df['mortality_bin'].astype('Int64')

    # ----------------------
    # 3. Treat county class as categorical
    # ----------------------
    df['county_class'] = df['county_class'].astype('category')

    return df, bin_edges_dict



def ordinal_mortality_classifier(df_cat, svi_vars, n_splits=5):
    """
    Train XGBoost classifier on binned mortality rates (technically not ordinal classification, but using a multi-class classifier to do ordinal classification).
    Performs k-fold CV per year and returns OOS predictions and metrics.
    """
    all_predictions = []
    all_metrics = []

    # Predict mortality_bin from year t+1 using year t's SVI
    for year in sorted(df_cat['year'].unique()):
        df_year = df_cat[df_cat['year'] == year].copy()

        if df_year.empty:
            print(f"⚠️ No data for year {year}")
            continue

        print(f"\n🔁 Year {year} → predicting mortality_bin for year {year+1}")

        # Features and target
        feature_cols = [f'{var}_bin' for var in svi_vars] + ['county_class']
        target_col = 'mortality_bin'

        X = df_year[feature_cols].copy()
        y = df_year[target_col].copy()
        fips = df_year['FIPS']

        # # Treat categorical properly
        # X['county_class'] = X['county_class'].astype("category")
        
        # Convert all features (not just urbanicity) to categorical dtype
        for col in X.columns:
            X[col] = X[col].astype("category")

        # K-Fold per year
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            fips_test = fips.iloc[test_idx]

            # model = xgb.XGBClassifier(
            #     objective='multi:softprob',
            #     num_class=10,  # classes 0–9
            #     #max_depth=6,
            #     max_depth=4,
            #     #learning_rate=0.05,
            #     learning_rate=0.10,
            #     #n_estimators=300,
            #     n_estimators=500,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     #use_label_encoder=False,
            #     eval_metric='mlogloss',
            #     enable_categorical=True,
            #     tree_method='hist',  # Fastest for categorical
            #     n_jobs=-1,
            #     random_state=42
            # )
            
            model = XGBOrdinal(
                    objective='binary:logistic',   # Each sub-model is binary
                    max_depth=4,
                    learning_rate=0.1,
                    n_estimators=300,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method='hist',
                    enable_categorical=True,
                    n_jobs=-1,
                    random_state=42,
                    aggregation='weighted',  # or 'equal' if you prefer uniform weighting
                    norm=True
                )
            
            ### 8/22/25, EB: Adding sample weights to handle class imbalance, because model is still performing poorly.
            # from sklearn.utils.class_weight import compute_sample_weight
            # sample_weights = compute_sample_weight('balanced', y_train)
            # model.fit(X_train, y_train, sample_weight=sample_weights)
            # model.fit(X_train, y_train)

            # # Get predictions
            # y_pred = model.predict(X_test)                    # hard prediction (argmax)
            # y_prob = model.predict_proba(X_test)              # soft probabilities
            # expected_bin = np.sum(y_prob * np.arange(y_prob.shape[1]), axis=1)  # expected bin
            
            # # 8/22/25, EB: Type error fix
            # y_test_int = y_test.astype(int)

            # # Store predictions
            # pred_df = pd.DataFrame({
            #     'FIPS': fips_test.values,
            #     'Year': year + 1,  # prediction year
            #     'TrueBin': y_test_int.values,
            #     'PredBin': y_pred,
            #     'ExpectedBin': expected_bin,
            #     'Fold': fold + 1
            # })
            # all_predictions.append(pred_df)

            # # Compute standard classification metrics
            # acc = accuracy_score(y_test_int, y_pred)
            # report = classification_report(y_test_int, y_pred, output_dict=True, zero_division=0)
            # mae = mean_absolute_error(y_test_int, expected_bin)

            # # Store metrics
            # all_metrics.append({
            #     'Year': year + 1,
            #     'Fold': fold + 1,
            #     'Accuracy': acc,
            #     'MacroF1': report['macro avg']['f1-score'],
            #     'WeightedF1': report['weighted avg']['f1-score'],
            #     'MAE': mae
            # })
#############################################################
            # # print("X_train missing values:\n", X_train.isna().sum())
            # # print("y_train missing values:", y_train.isna().sum())
            # # print("X_test missing values:\n", X_test.isna().sum())
            # # print("y_test missing values:", y_test.isna().sum())

            # model.fit(X_train, y_train)

            # # Predict class (hard prediction) and probability (soft prediction)
            # y_pred = model.predict(X_test)                    # hard class prediction
            # y_prob = model.predict_proba(X_test)              # soft probability distribution
            # expected_bin = np.sum(y_prob * np.arange(y_prob.shape[1]), axis=1)  # expected bin (fuzzy)
            # print("✅ y_pred contains NaNs:", pd.isna(y_pred).any())
            # print("✅ y_prob contains NaNs:", np.isnan(y_prob).any())
            # print("✅ Any row in y_prob all zero?:", np.any(np.all(y_prob == 0, axis=1)))
            # print("✅ Shape of y_prob:", y_prob.shape)

            # # Sanitize y_test (drop NaNs and make sure type is int)
            # y_test_clean = y_test.dropna()
            # y_pred_series = pd.Series(y_pred, index=X_test.index)
            # y_pred_clean = y_pred_series.loc[y_test_clean.index]

            # # Ensure both are integer arrays (avoid nullable pandas types)
            # y_test_clean = y_test_clean.astype(int)
            # y_pred_clean = y_pred_clean.astype(int)

            # # Store predictions (only for valid index subset)
            # pred_df = pd.DataFrame({
            #     'FIPS': fips_test.loc[y_test_clean.index].values,
            #     'Year': year + 1,
            #     'TrueBin': y_test_clean.values,
            #     'PredBin': y_pred_clean.values,
            #     'ExpectedBin': expected_bin[y_test_clean.index],
            #     'Fold': fold + 1
            # })
            # all_predictions.append(pred_df)

            # # Compute metrics
            # acc = accuracy_score(y_test_clean, y_pred_clean)
            # report = classification_report(y_test_clean, y_pred_clean, output_dict=True, zero_division=0)
            # mae = mean_absolute_error(y_test_clean, expected_bin[y_test_clean.index])

            # # Store metrics
            # all_metrics.append({
            #     'Year': year + 1,
            #     'Fold': fold + 1,
            #     'Accuracy': acc,
            #     'MacroF1': report['macro avg']['f1-score'],
            #     'WeightedF1': report['weighted avg']['f1-score'],
            #     'MAE': mae
            # })


##################################################
            # # Fit the model
            # model.fit(X_train, y_train)

            # # Get predictions
            # y_pred = model.predict(X_test)                    # hard prediction (argmax)
            # ### Troubleshooting the y_pred type. We get dtype 'object' from the model, but all of our performance metrics need numpy arrays.
            # y_pred = pd.Series(y_pred).astype(int).values
            
            # y_prob = model.predict_proba(X_test)              # soft probabilities
            # expected_bin = np.sum(y_prob * np.arange(y_prob.shape[1]), axis=1)  # expected bin (float)

            # # Reset indices for alignment
            # y_test_int = y_test.reset_index(drop=True).astype(int)
            # fips_test_reset = fips_test.reset_index(drop=True)

            # # Store predictions
            # pred_df = pd.DataFrame({
            #     'FIPS': fips_test_reset,
            #     'Year': year + 1,  # prediction year
            #     'TrueBin': y_test_int,
            #     'PredBin': y_pred,
            #     'ExpectedBin': expected_bin,
            #     'Fold': fold + 1
            # })
            # all_predictions.append(pred_df)

            # print("🧪 y_test_int dtype:", y_test_int.dtype)
            # print("🧪 y_pred dtype:", y_pred.dtype)
            # print("🧪 y_test_int unique values:", np.unique(y_test_int))
            # print("🧪 y_pred unique values:", np.unique(y_pred))
            # print("🧪 y_test_int shape:", y_test_int.shape)
            # print("🧪 y_pred shape:", y_pred.shape)
            # from sklearn.utils.multiclass import type_of_target
            # print("🧪 type of y_test_int:", type_of_target(y_test_int))
            # print("🧪 type of y_pred:", type_of_target(y_pred))


            # # Compute metrics
            # acc = accuracy_score(y_test_int, y_pred)
            # report = classification_report(y_test_int, y_pred, output_dict=True, zero_division=0)
            # mae = mean_absolute_error(y_test_int, expected_bin)

            # # Store metrics
            # all_metrics.append({
            #     'Year': year + 1,
            #     'Fold': fold + 1,
            #     'Accuracy': acc,
            #     'MacroF1': report['macro avg']['f1-score'],
            #     'WeightedF1': report['weighted avg']['f1-score'],
            #     'MAE': mae
            # })
########################################################
            # Train the model
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            #model.fit(X_train, y_train)

            # Get predictions
            y_pred = model.predict(X_test)                    # hard prediction (argmax)
            y_pred = pd.Series(y_pred).astype(int).values     # 👈 ensure correct dtype

            y_prob = model.predict_proba(X_test)              # soft probabilities
            expected_bin = np.sum(y_prob * np.arange(y_prob.shape[1]), axis=1)  # expected bin

            # Store predictions
            pred_df = pd.DataFrame({
                'FIPS': fips_test.values,
                'Year': year + 1,  # prediction year
                'TrueBin': y_test.values,
                'PredBin': y_pred,
                'ExpectedBin': expected_bin,
                'Fold': fold + 1
            })
            all_predictions.append(pred_df)

            # Compute standard classification metrics
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            mae = mean_absolute_error(y_test, expected_bin)

            # Store metrics
            all_metrics.append({
                'Year': year + 1,
                'Fold': fold + 1,
                'Accuracy': acc,
                'MacroF1': report['macro avg']['f1-score'],
                'WeightedF1': report['weighted avg']['f1-score'],
                'MAE': mae
            })

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    return predictions_df, metrics_df

def performance_evaluation(metrics_df, predictions_df):
    """
    Summarize overall and per-year performance for ordinal classification, designed for the XGBoost ordinal model.
    """

    # Summary across folds per year
    year_summary = metrics_df.groupby("Year").agg({
        "Accuracy": "mean",
        "MacroF1": "mean",
        "WeightedF1": "mean",
        "MAE": "mean"
    }).reset_index()

    print("\n📈 Evaluation Results (Average over Folds):\n")
    print(year_summary.round(4))

    # Overall MAE from expected bin
    overall_mae = mean_absolute_error(predictions_df['TrueBin'], predictions_df['ExpectedBin'])
    print(f"\n📉 Overall MAE (Expected Bin vs True Bin): {overall_mae:.4f}")

    return year_summary, overall_mae



def evaluate_ordinal_classification(y_true, y_pred, class_labels=None, save_dir=None, prefix="overall", show_plot=False):
    """
    Evaluates ordinal classification performance with confusion matrix, weighted heatmap, and MAE.
    """
    # Default labels for 10 ordinal classes
    if class_labels is None:
        class_labels = list(range(10))

    # 1. Standard confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{prefix.capitalize()} Confusion Matrix")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{prefix}_confusion_matrix.png", dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    # 2. Weighted heatmap (|i - j| as penalty)
    penalty_matrix = np.abs(np.subtract.outer(class_labels, class_labels))
    weighted_cm = cm * penalty_matrix

    plt.figure(figsize=(8, 6))
    sns.heatmap(weighted_cm, annot=True, fmt='.0f', cmap='coolwarm', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"{prefix.capitalize()} Weighted Ordinal Confusion Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{prefix}_weighted_heatmap.png", dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    # 3. Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    print(f"📊 {prefix.capitalize()} Mean Absolute Error (MAE): {mae:.4f}")

    return {
        "confusion_matrix": cm,
        "weighted_cm": weighted_cm,
        "mae": mae
    }

#######
### Function to check bin counts, for peace of mind.
def inspect_mortality_bin_counts(df):
    """
    Prints the number of counties in each mortality_bin (0–9) for every year.
    """
    print("🧮 Mortality Bin Counts by Year:\n")

    all_years = sorted(df['year'].unique())
    all_bins = range(10)

    for year in all_years:
        year_df = df[df['year'] == year]
        counts = year_df['mortality_bin'].value_counts(dropna=False).sort_index()

        print(f"📅 Year {year}:")
        for b in all_bins:
            count = counts.get(b, 0)
            label = f"Bin {b} (zero)" if b == 0 else f"Bin {b}"
            print(f"   {label}: {count} counties")
        print()

#######
### Function to tune hyperparameters for the XGBoost model
def tune_xgb_classifier_params(df_cat, svi_vars, param_grid, n_splits=5):
    """
    Grid search over XGBoost hyperparameters for ordinal classification.
    Returns a DataFrame of average performance metrics per config.
    """
    results = []
    param_names = list(param_grid.keys())
    param_combos = list(product(*param_grid.values()))

    for combo in tqdm(param_combos, desc="Tuning Hyperparameters"):
        param_dict = dict(zip(param_names, combo))

        all_y_true = []
        all_y_pred = []

        for year in sorted(df_cat['year'].unique()):
            df_year = df_cat[df_cat['year'] == year].copy()

            if df_year.empty:
                continue

            feature_cols = [f'{var}_bin' for var in svi_vars] + ['county_class']
            target_col = 'mortality_bin'

            X = df_year[feature_cols].copy()
            y = df_year[target_col].copy()

            # Ensure all features are categorical
            for col in X.columns:
                X[col] = X[col].astype("category")
            #y = y.astype("category")

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=10,
                    enable_categorical=True,
                    tree_method='hist',
                    n_jobs=-1,
                    random_state=42,
                    **param_dict
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                all_y_true.extend(y_test.astype(int))
                all_y_pred.extend(y_pred.astype(int))

        mae = mean_absolute_error(all_y_true, all_y_pred)
        acc = accuracy_score(all_y_true, all_y_pred)
        report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)

        results.append({
            **param_dict,
            "MAE": mae,
            "Accuracy": acc,
            "MacroF1": report['macro avg']['f1-score'],
            "WeightedF1": report['weighted avg']['f1-score']
        })

    results_df = pd.DataFrame(results).sort_values("MAE")
    return results_df

#######
### Function to look at per-year performance.
def evaluate_per_year(predictions_df):
    """
    Evaluate model performance per year:
    - Confusion matrix
    - Accuracy
    - Class-wise precision/recall/F1
    """

    years = predictions_df['Year'].unique()
    results = []

    for year in sorted(years):
        df_year = predictions_df[predictions_df['Year'] == year]
        y_true = df_year['TrueBin']
        y_pred = df_year['PredBin']

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=range(10))

        print(f"\n📅 Year {year}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)

        results.append({
            'Year': year,
            'Accuracy': acc,
            'MacroF1': report['macro avg']['f1-score'],
            'WeightedF1': report['weighted avg']['f1-score']
        })

    results_df = pd.DataFrame(results)
    return results_df

#######
### Function to evaluate several metrics, by year
def performance_evaluation_v2(predictions_df, metrics_df=None, save_plots=False, output_dir="./evaluation_plots"):
    """
    Extended evaluation function for ordinal classification.
    Includes:
    - Metrics by year (Accuracy, F1s, MAE)
    - Confusion matrices per year
    - Plots: ExpectedBin vs TrueBin and MAE over time
    """

    print("\n📊 Evaluating Ordinal Classification Performance...\n")

    # ----------------------------------------
    # 1. Metrics by year (recompute if needed)
    # ----------------------------------------
    if metrics_df is None:
        metrics_df = predictions_df.groupby("Year").apply(
            lambda df: pd.Series({
                "Accuracy": (df["TrueBin"] == df["PredBin"]).mean(),
                "MAE": mean_absolute_error(df["TrueBin"], df["ExpectedBin"])
            })
        ).reset_index()

    print("📈 Metrics by Year (recomputed from predictions):")
    print(metrics_df.round(4))

    # ----------------------------------------
    # 2. Plot: ExpectedBin vs TrueBin (scatter)
    # ----------------------------------------
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=predictions_df, x="TrueBin", y="ExpectedBin", alpha=0.3, edgecolor=None)
    plt.plot([0, 9], [0, 9], ls='--', color='gray')
    plt.title("Expected Bin vs True Bin")
    plt.xlabel("True Bin")
    plt.ylabel("Expected Bin")
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{output_dir}/expected_vs_true_bin.png", bbox_inches="tight")
    plt.show()

    # ----------------------------------------
    # 3. Plot: MAE over Time
    # ----------------------------------------
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=metrics_df, x="Year", y="MAE", marker="o")
    plt.title("Mean Absolute Error Over Time")
    plt.ylabel("MAE (ExpectedBin vs TrueBin)")
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{output_dir}/mae_over_time.png", bbox_inches="tight")
    plt.show()

    # ----------------------------------------
    # 4. Confusion Matrices by Year
    # ----------------------------------------
    unique_years = sorted(predictions_df['Year'].unique())
    for year in unique_years:
        year_df = predictions_df[predictions_df['Year'] == year]
        y_true = year_df['TrueBin'].astype(int)
        y_pred = year_df['PredBin'].astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
        disp.plot(cmap='Blues', xticks_rotation=45, values_format='.0f')
        plt.title(f"Confusion Matrix for Year {year}")
        if save_plots:
            plt.savefig(f"{output_dir}/confusion_matrix_{year}.png", bbox_inches="tight")
        plt.show()

    # ----------------------------------------
    # 5. Return summary
    # ----------------------------------------
    overall_mae = mean_absolute_error(predictions_df['TrueBin'], predictions_df['ExpectedBin'])
    print(f"\n📉 Overall MAE (Expected Bin vs True Bin): {overall_mae:.4f}")

    return metrics_df, overall_mae



def main():
    # Load and prepare data
    df = prepare_yearly_prediction_data()
    
    # Define SVI variables
    svi_vars = [v for v in DATA if v != 'Mortality']
    
    # Transform to categorical format
    df_cat, bin_edges_dict = transform_to_categorical(df, svi_vars)
    # inspect_mortality_bin_counts(df_cat)  # Check bin counts for peace of mind
    
    # Run ordinal classification
    predictions_df, metrics_df = ordinal_mortality_classifier(df_cat, svi_vars, n_splits=5)
    
    # Look at per-year performance
    _, overall_mae = performance_evaluation_v2(predictions_df, metrics_df, save_plots=False)
    
    print("Evaluation Results:")
    print('')
    print(overall_mae)



# # Tune hyperparameters
# def main():
#     df = prepare_yearly_prediction_data()
#     svi_vars = [v for v in DATA if v != 'Mortality']
#     df_cat, bin_edges_dict = transform_to_categorical(df, svi_vars)
#     param_grid = {
#                     "max_depth": [4],#, 6, 8],
#                     "learning_rate": [0.10],#[0.01, 0.05, 0.1],
#                     "n_estimators": [100, 200, 300, 500],
#                     "subsample": [0.8],
#                     "colsample_bytree": [0.8]
#                 }

#     results_df = tune_xgb_classifier_params(df_cat, svi_vars, param_grid, n_splits=5)
#     print(results_df)


if __name__ == "__main__":
    main()