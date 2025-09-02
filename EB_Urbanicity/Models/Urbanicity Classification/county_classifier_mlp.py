### 3/31/25, EB: Here I am trying to implement an MLP classifier using TensorFlow/Keras to classify counties based on yearly SVI data.
### I tried using XGB, but it wasn't able to effectively handle the yearly aspect of the data. I tried using summary stats, but the prediction precision
### was very poor, worse than a coin flip. I'm hoping that an MLP will be able to learn the patterns in the data better than XGB.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# def prepare_temporal_data():
#     """Loads and reshapes data into a temporal format (n_counties, n_years, n_features)."""
#     # Load all variables for all years (adjust paths as needed)
#     all_data = []
#     variables = [v for v in DATA if v not in ['Mortality']]
    
#     for variable in variables:
#         var_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
#         var_df = pd.read_csv(var_path, dtype={'FIPS': str})
#         var_df['FIPS'] = var_df['FIPS'].str.zfill(5)
        
#         # Extract yearly columns (e.g., "2010 Unemployment", "2011 Unemployment"...)
#         yearly_cols = [f'{year} {variable}' for year in range(2010, 2023)]
#         var_data = var_df[['FIPS'] + yearly_cols].set_index('FIPS')
#         all_data.append(var_data)
    
#     # Combine all variables into a single DataFrame
#     combined_df = pd.concat(all_data, axis=1)
    
#     # Load urban-rural labels
#     urban_rural = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
#     urban_rural['FIPS'] = urban_rural['FIPS'].str.zfill(5).set_index('FIPS')
#     combined_df = combined_df.join(urban_rural['2023 Code'], how='inner')
#     combined_df['urban_rural_class'] = combined_df['2023 Code'].astype(int) - 1
#     combined_df = combined_df.drop(columns=['2023 Code'])
    
#     # Reshape into (n_counties, n_years * n_features)
#     # Each row is a county, with columns ordered as [Year1_Var1, Year1_Var2, ..., Year13_VarN]
#     n_years = 13  # 2010-2022
#     n_features = len(variables)
#     X = combined_df.drop(columns=['urban_rural_class']).values.reshape(-1, n_years * n_features)
#     y = combined_df['urban_rural_class'].values
    
#     return X, y

# # Generate data
# X, y = prepare_temporal_data()


# # Split into train/test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Standardize features (critical for MLPs)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# def build_mlp(input_shape, n_classes):
#     model = Sequential([
#         Dense(256, activation='relu', input_shape=(input_shape,)),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(128, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         Dense(n_classes, activation='softmax')  # Use 'sigmoid' for binary classification
#     ])
    
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss='sparse_categorical_crossentropy',  # Use 'binary_crossentropy' for 2 classes
#         metrics=['accuracy']
#     )
#     return model

# # Initialize model
# n_classes = len(np.unique(y))
# model = build_mlp(X_train_scaled.shape[1], n_classes)
# model.summary()


# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# history = model.fit(
#     X_train_scaled, y_train,
#     validation_split=0.2,
#     epochs=100,
#     batch_size=32,
#     callbacks=[early_stop],
#     class_weight='balanced'  # Handles class imbalance
# )


# # Evaluate on test set
# test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
# print(f"Test Accuracy: {test_acc:.3f}")

# # Generate predictions
# y_pred = model.predict(X_test_scaled).argmax(axis=1)  # For multi-class

# # Classification report
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))







###########################################################################################
### 3/31/25, EB: Trying to install Tensorflow to use the above code, and it's taking forever.
### I had DS refactor the above to be used in a PyTorch format, and we'll see how it works here.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight


# ----------------------
# 1. Data Preparation (Same as before)
# ----------------------

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


# Generate data
X, y = prepare_temporal_data()
n_classes = len(np.unique(y))

# ----------------------
# 2. PyTorch Dataset Class
# ----------------------
class CountyDataset(Dataset):
    def __init__(self, features, labels):
        # Convert to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ----------------------
# 3. Preprocessing & DataLoaders
# ----------------------
# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create DataLoaders
train_dataset = CountyDataset(X_train_scaled, y_train)
test_dataset = CountyDataset(X_test_scaled, y_test)

batch_size = 32
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------
# 4. MLP Model Definition
# ----------------------
# class MLP(nn.Module):
#     def __init__(self, input_size, num_classes=2):
#         super(MLP, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
            
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
            
#             nn.Linear(128, 64),
#             nn.ReLU(),
            
#             nn.Linear(64, num_classes)
#         )
        
#     def forward(self, x):
#         return self.layers(x)

class EnhancedMLP(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            # Wider input layer
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Additional hidden layers
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # Final output layer
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

# ----------------------
# 5. Training Setup
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedMLP(X_train_scaled.shape[1], n_classes).to(device)

# Handle class imbalance
class_weights = torch.tensor(
    compute_class_weight('balanced', classes=np.unique(y), y=y),
    dtype=torch.float32
).to(device)

sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))


# Modify your train_loader creation
train_loader = DataLoader(train_dataset, batch_size=32, 
                         sampler=sampler, shuffle=False)  # shuffle=False when using sampler

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------
# 6. Training Loop with Early Stopping
# ----------------------
# def train_model(model, train_loader, test_loader, epochs=501, patience=10):
#     best_loss = float('inf')
#     patience_counter = 0
    
#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         train_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item() * inputs.size(0)
        
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() * inputs.size(0)
        
#         # Calculate metrics
#         train_loss = train_loss / len(train_loader.dataset)
#         val_loss = val_loss / len(test_loader.dataset)
        
#         if epoch % 50 == 0:
#             print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
#         # Early stopping
#         if val_loss < best_loss:
#             best_loss = val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), 'best_model.pth')
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print("Early stopping!")
#                 break

#     # Load best model
#     model.load_state_dict(torch.load('best_model.pth'))
#     return model

def train_model(model, train_loader, test_loader, epochs=501, patience=100):
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=50, 
                                                    verbose=True)
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Store losses for monitoring
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Class-balanced sampling setup (do this once before training loop)
        # Move this outside the function to avoid recomputing every epoch
        # class_weights = compute_class_weight('balanced', 
        #                    classes=np.unique(y_train), y=y_train)
        # sample_weights = class_weights[y_train]
        # sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate metrics
        train_loss /= len(train_loader.dataset)
        val_loss /= len(test_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        if epoch % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs} | LR: {current_lr:.1e} | '
                  f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break

    # Load best model weights
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_losses, val_losses


# Start training
model, train_losses, val_losses = train_model(model, train_loader, test_loader)

# ----------------------
# 7. Evaluation
# ----------------------
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print(classification_report(all_labels, all_preds))

evaluate_model(model, test_loader)

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()