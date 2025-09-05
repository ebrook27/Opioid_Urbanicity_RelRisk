### 8/21/25, EB: Ok. Talked to Andrew today and he made a lot of good points and suggestions. I wasn't treating county category in the best way, but now that I've got it
### as a pure categorical variable, and I know XGBoost can handle it well, I think that opens us up to a few new options.
### What I'm going to try here is to convert the problem to a purely categorical one. By that I mean, bin up all the data, SVI and mortality (urbanicity is already),
### into 10% deciles, and then use an XGBoost model to try to predict the decile of mortality for each county in each year.
### If this works directly, then great, but if not, we can try to define a custom loss function that turns XGBoost into an ordinal regression model.
### To bin up the mortality data, Andrew said that they're thinking we put the zero mortality counties into the lowest decile, and then bin the rest of the counties
### using a log-normal distribution. For the SVI data, it's already percentile ranks, so we can take 10% slices directly.
### This will give us all categorical data, which should be more straightforward for XGBoost to handle. We'll give it a shot at least.


### 8/22/25, EB: Alright, so this is working, and I tuned the hyperparameters, and I weighted the classes to handle the class imbalance. This is still not working well. Accuracy is up to 0.41-ish in the best year,
### and the other scores are all worse. If we look at the confusion matrix, class 0 is way over-predicted, and the other classes are all over the place. For some mortality bins, some years, the model isn't even predicting the correct
### bin the most. There's a lot of bleed over into neighboring classes.
### I found a paper last night that describes a simple method for turning a multi-class classifier into an ordinal regression model, and it seems like it would be easy to implement. We perform several binary classifications, one for each cut point between the bins,
### and then combine the results to get the final probability distribution over the bins. There's an implementation of this to XGBoost classifiers in the `xgboost-ordinal` package, so I'm going to try that next.
### If this doesn't help, I think we gotta think of something else to try. 



import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
from tqdm import tqdm
from itertools import product

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
        # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # for f, (_, test_idx) in enumerate(kf.split(X), 1):
        #     print(f'Fold {f} y distribution:\n', y.iloc[test_idx].value_counts(normalize=True).sort_index())

        # for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):       
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            fips_test = fips.iloc[test_idx]

            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=10,  # classes 0–9
                #max_depth=6,
                max_depth=4,
                #learning_rate=0.05,
                learning_rate=0.10,
                #n_estimators=300,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                #use_label_encoder=False,
                eval_metric='mlogloss',
                enable_categorical=True,
                tree_method='hist',  # Fastest for categorical
                n_jobs=-1,
                random_state=42
            )
            
            ### 8/22/25, EB: Adding sample weights to handle class imbalance, because model is still performing poorly.
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            #model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Store predictions
            pred_df = pd.DataFrame({
                'FIPS': fips_test.values,
                'Year': year + 1,  # prediction year
                'TrueBin': y_test.values,
                'PredBin': y_pred,
                'Fold': fold + 1
            })
            all_predictions.append(pred_df)

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            all_metrics.append({
                'Year': year + 1,
                'Fold': fold + 1,
                'Accuracy': acc,
                'MacroF1': report['macro avg']['f1-score'],
                'WeightedF1': report['weighted avg']['f1-score']
            })

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    return predictions_df, metrics_df


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
    
    # # Evaluate overall performance
    # overall_eval = evaluate_ordinal_classification(
    #     predictions_df['TrueBin'],
    #     predictions_df['PredBin'],
    #     class_labels=list(range(10)),
    #     #save_dir="County Classification/Ordinal_Eval_Plots",
    #     #prefix="overall",
    #     show_plot=True
    # )
    
    # Look at per-year performance
    overall_eval = evaluate_per_year(predictions_df)
    
    print("Evaluation Results:")
    print('')
    print(overall_eval)



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