### 8/18/25, EB: I talked with Andrew last week, and we threw around several ideas for how to predict relative risk. I had been trying to use an LSTM
### to predict relative risk deciles, but it was doing so bad, even when I separated out the zero-mortality counties (up to 30% of the counties in some years),
### so he suggested I try some sort of logistic regression model. Here I'm going to try a simple logistic regression model to predict whether a county is in the top 10% of mortality rates for that year.

### 8/18/25, EB: I added a few things to try to improve the model performance: did a CV to get OOS predictions, used a threshold of 0.72 to classify high-risk counties (by looking at the precision-recall curve),
### and used class_weight='balanced' to help with the class imbalance. The performance improved, but the model is still doing a bad job of predicting high-risk counties. I'm going to try using different "high-risk"
### thresholds, like 0.8 or 0.9, to see if that helps, then move on to an ordinal logistic regression model.

######################
### 8/19/25, EB: Ok I realized this was pretty wrong, I wasn't doing things year-by-year. This was mixing all of the data up together, which was not what we are looking for. I think we might be able to do a growing window model, but for now I'm going to just do a yearly model.


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt




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

def label_high_risk(df, quantile=0.9):
    df = df.copy()
    threshold = df['mortality_rate'].quantile(quantile)
    df['high_risk'] = (df['mortality_rate'] >= threshold).astype(int)
    return df


# def run_logistic_regression(data_df):
#     features = [col for col in DATA if col != 'Mortality'] + ['county_class']
#     target = 'high_risk'

#     X = data_df[features]
#     y = data_df[target]

#     # Split into train/test
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, stratify=y, test_size=0.2, random_state=42
#     )

#     # Preprocess county_class
#     preprocessor = ColumnTransformer([
#         ('cat', OneHotEncoder(drop='first'), ['county_class'])
#     ], remainder='passthrough')

#     # Pipeline
#     pipeline = Pipeline([
#         ('prep', preprocessor),
#         ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
#     ])

#     # Fit and predict
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     y_prob = pipeline.predict_proba(X_test)[:, 1]

#     # Evaluate
#     print("🔍 Classification Report:\n", classification_report(y_test, y_pred))
#     print("🔵 ROC AUC:", roc_auc_score(y_test, y_prob))
    
#     return pipeline


def run_logistic_regression_with_cv(data_df, n_splits=5):
    features = [col for col in DATA if col != 'Mortality'] + ['county_class']
    target = 'high_risk'

    X = data_df[features].copy()
    y = data_df[target].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_true = []
    all_pred = []
    all_prob = []
    all_index = []
    ### 8/18/25, EB: Plotted the precision-recall curve to determine the best threshold for classification.
    threshold = 0.72  # You can choose this based on your PR curve

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), ['county_class'])
        ], remainder='passthrough')

        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(max_iter=1500, class_weight='balanced', solver='lbfgs'))
        ])

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        #y_pred = pipeline.predict(X_test)
        y_pred = (y_prob >= threshold).astype(int)  # Apply threshold

        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_prob.extend(y_prob)
        all_index.extend(X_test.index)

    # Final evaluation
    print("🔍 Classification Report:\n", classification_report(all_true, all_pred, target_names=["low risk", "high risk"]))
    print("🔵 ROC AUC:", roc_auc_score(all_true, all_prob))

    # ### 8/18/25, EB: Performance improved greatly with the class_weight='balanced' parameter. Now trying to deterimine the best threshold for classification.
    # # Compute precision-recall pairs for different probability thresholds
    # precisions, recalls, thresholds = precision_recall_curve(all_true, all_prob)

    # # Plot Precision-Recall vs Threshold
    # plt.figure(figsize=(10, 6))
    # plt.plot(thresholds, precisions[:-1], label="Precision", linewidth=2)
    # plt.plot(thresholds, recalls[:-1], label="Recall", linewidth=2)
    # plt.xlabel("Decision Threshold")
    # plt.ylabel("Score")
    # plt.title("Precision and Recall vs Threshold")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # Build prediction DataFrame
    results_df = data_df.loc[all_index].copy()
    results_df['True'] = all_true
    results_df['Predicted'] = all_pred
    results_df['Probability'] = all_prob

    return results_df


def run_temporal_kfold_logreg(data_df, n_splits=5, threshold=0.72):
    '''
    Runs logistic regression with temporal k-fold CV, predicting next-year high-risk status using current-year features.
    '''
    
    features = [col for col in DATA if col != 'Mortality'] + ['county_class']
    target = 'high_risk'
    
    years = sorted(data_df['year'].unique())
    all_true, all_pred, all_prob, all_index = [], [], [], []

    for year in years:
        train_df = data_df[data_df['year'] == year].copy()
        test_df = data_df[data_df['year'] == year].copy()

        if test_df.empty:
            continue  # skip final year

        X = train_df[features]
        y = test_df[target]  # target is for year+1
        if len(X) != len(y):
            # Align by county if needed (e.g., merge on FIPS)
            merged = train_df[['FIPS'] + features].merge(
                test_df[['FIPS', target]], on='FIPS'
            )
            X = merged[features]
            y = merged[target]
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['county_class'])
            ], remainder='passthrough')

            pipeline = Pipeline([
                ('prep', preprocessor),
                ('clf', LogisticRegression(max_iter=1500, class_weight='balanced'))
            ])

            pipeline.fit(X_train, y.iloc[train_idx])
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            all_true.extend(y_val)
            all_pred.extend(y_pred)
            all_prob.extend(y_prob)
            all_index.extend(X_val.index)

        print(f"✅ Completed temporal CV for year {year} ➝ {year+1}")

    # Final evaluation
    print("\n🔍 Classification Report:\n", classification_report(all_true, all_pred, target_names=["low risk", "high risk"]))
    print("🔵 ROC AUC:", roc_auc_score(all_true, all_prob))

    results_df = data_df.loc[all_index].copy()
    results_df['True'] = all_true
    results_df['Predicted'] = all_pred
    results_df['Probability'] = all_prob

    return results_df

def run_logistic_regression_expanding_window(data_df, start_year=2010, end_year=2021):
    '''
    Runs logistic regression with an expanding window approach, predicting next-year high-risk status using all prior years' data.
    '''    
    
    features = [col for col in DATA if col != 'Mortality'] + ['county_class']
    target = 'high_risk'

    all_true = []
    all_pred = []
    all_prob = []
    all_index = []

    threshold = 0.72  # Set based on prior PR curve

    for year in range(start_year, end_year):
        # TRAIN on data from years <= current year
        train_df = data_df[data_df['year'] <= year].copy()
        test_df = data_df[data_df['year'] == year].copy()

        if train_df.empty or test_df.empty:
            continue  # Skip if any year is missing data

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        # Preprocessing
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), ['county_class'])
        ], remainder='passthrough')

        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(max_iter=1500, class_weight='balanced', solver='lbfgs'))
        ])

        # Fit model
        pipeline.fit(X_train, y_train)

        # Predict and apply threshold
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # Store results
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_prob.extend(y_prob)
        all_index.extend(X_test.index)

        print(f"✅ Completed prediction for year {year + 1} using training data through {year}.")

    # Evaluation
    print("🔍 Final Classification Report:\n", classification_report(all_true, all_pred, target_names=["low risk", "high risk"]))
    print("🔵 ROC AUC:", roc_auc_score(all_true, all_prob))

    # Return result DataFrame
    results_df = data_df.loc[all_index].copy()
    results_df['True'] = all_true
    results_df['Predicted'] = all_pred
    results_df['Probability'] = all_prob

    return results_df



######################################################################################

def main():
    # Prepare data
    df = prepare_yearly_prediction_data()
    
    # Label high-risk counties
    df = label_high_risk(df, quantile=0.9)

    # Run logistic regression
    results_df = run_logistic_regression_with_cv(df)

# def main():
#     from sklearn.metrics import roc_auc_score, f1_score
#     # Prepare data
#     data_df = prepare_yearly_prediction_data()
    
#     thresholds = [0.99, 0.98, 0.95, 0.90, 0.85, 0.80, 0.75]  # Adjust as needed (top 5%, 10%, etc.)
#     results_summary = []

#     for t in thresholds:
#         # Create new binary target column based on threshold
#         cutoff = data_df['mortality_rate'].quantile(t)
#         data_df['high_risk'] = (data_df['mortality_rate'] >= cutoff).astype(int)

#         print(f"\n🧪 Evaluating for threshold: top {(1 - t) * 100:.0f}% counties as high risk")
#         results_df = run_logistic_regression_with_cv(data_df)

#         # Optional: compute and store additional metrics
#         roc_auc = roc_auc_score(results_df['True'], results_df['Probability'])
#         f1 = f1_score(results_df['True'], results_df['Predicted'])

#         results_summary.append({
#             'Threshold': f'top {(1 - t) * 100:.0f}%',
#             'ROC AUC': roc_auc,
#             'F1 (high risk)': f1,
#         })

#     # Display summary
#     import pandas as pd
#     summary_df = pd.DataFrame(results_summary)
#     print("\n🔍 Summary of performance across thresholds:\n", summary_df)


# def main():
#     # Prepare data
#     df = prepare_yearly_prediction_data()
    
#     # Label high-risk counties
#     df = label_high_risk(df, quantile=0.9)

#     # Run logistic regression
#     results_df = run_logistic_regression_expanding_window(df)

if __name__ == "__main__":
    main()