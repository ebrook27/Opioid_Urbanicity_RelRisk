### 4/3/25, EB: Talked to VM, he suggested trying a Logistic Regression model, and then trying a yearly prediciton.
### Here I'm going to first try the Logistic Regression model, and then I'll try the yearly prediction.

import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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
    
    ### 3/31/25, EB: I changed this to be 0-1 instead of 0-5, so that it is binary.
    ### This smakes it into an urban vs rural classification, instead of all 6 classes.
    
    combined_df['urban_rural_class'] = (combined_df['2023 Code'] >= 5).astype(int) #combined_df['2023 Code'].astype(int) - 1
    combined_df = combined_df.drop(columns=['2023 Code'])
    
    # Reshape into (n_counties, n_years * n_features)
    # Each row is a county, with columns ordered as [Year1_Var1, Year1_Var2, ..., Year13_VarN]
    n_years = 13  # 2010-2022
    n_features = len(variables)
    X = combined_df.drop(columns=['urban_rural_class']).values.reshape(-1, n_years * n_features)
    y = combined_df['urban_rural_class'].values
    
    return X, y

# Call your data loader
X, y = prepare_temporal_data()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Optional: Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Urban', 'Rural'], yticklabels=['Urban', 'Rural'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Urban vs Rural County Prediction")
plt.show()