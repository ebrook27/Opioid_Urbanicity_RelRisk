### 3/31/25, EB: Couldn't really get the MLP working that well. It did ok with a binary classification, but one of the classes only had a precision of 0.69.
### I've used SVMs before for regression, and they did the best for that problem, (temp prediction in AFSD), and I know they can be used for classification as well.
### This might not work well, but I wanted to give it a quick shot.

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']

def prepare_temporal_data():
    """Loads and reshapes data into a temporal format (n_counties, n_years, n_features)."""
    # Load all variables for all years (adjust paths as needed)
    all_data = []
    variables = [v for v in DATA if v not in ['Mortality']]
    
    for variable in variables:
        var_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
        var_df = pd.read_csv(var_path, dtype={'FIPS': str})
        var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
        
        # Extract yearly columns (e.g., "2010 Unemployment", "2011 Unemployment"...)
        yearly_cols = [f'{year} {variable}' for year in range(2010, 2023)]
        var_data = var_df[['FIPS'] + yearly_cols].set_index('FIPS')
        all_data.append(var_data)
    
    # Combine all variables into a single DataFrame
    combined_df = pd.concat(all_data, axis=1)
    
    # Load urban-rural labels
    urban_rural = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
    urban_rural['FIPS'] = urban_rural['FIPS'].str.zfill(5)
    urban_rural = urban_rural.set_index('FIPS')
    
    combined_df = combined_df.join(urban_rural['2023 Code'], how='inner')
    combined_df['urban_rural_class'] = (combined_df['2023 Code'] >= 5).astype(int) #combined_df['2023 Code'].astype(int) - 1 #
    combined_df = combined_df.drop(columns=['2023 Code'])
    
    # Reshape into (n_counties, n_years * n_features)
    # Each row is a county, with columns ordered as [Year1_Var1, Year1_Var2, ..., Year13_VarN]
    n_years = 13  # 2010-2022
    n_features = len(variables)
    X = combined_df.drop(columns=['urban_rural_class']).values.reshape(-1, n_years * n_features)
    y = combined_df['urban_rural_class'].values
    
    return X, y


# Generate data
X, y = prepare_temporal_data()
n_classes = len(np.unique(y))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Feature scaling (critical for SVMs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# smote = SMOTE(random_state=42, k_neighbors=5)
# X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Initialize SVM with basic parameters
svm = SVC(
    kernel='rbf', 
    class_weight='balanced',  # Helps with class imbalance
    probability=True,         # For confidence scores
    random_state=42
)

# Hyperparameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=svm, 
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='f1_weighted'
)

grid_search.fit(X_train_scaled, y_train)

# Best model
best_svm = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluation
y_pred = best_svm.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  