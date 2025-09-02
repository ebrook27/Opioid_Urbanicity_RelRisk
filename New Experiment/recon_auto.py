import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
import random

# Constants
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]
LOSS_FUNCTION = nn.L1Loss()  # PyTorch's built-in loss function for MAE
DATA_YEARS = range(2010, 2023)
NUM_COUNTIES = 3144
NUM_EPOCHS = 100

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set up logging
log_file = 'New Experiment/auto.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def construct_mort_df(mort_path, mort_names):
    mort_df = pd.read_csv(mort_path, header=0, names=mort_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[mort_names[1:]] = mort_df[mort_names[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

class Tensors(Dataset):
    def __init__(self, data_df, mort_df, years=DATA_YEARS):
        self.data_df = data_df
        self.mort_df = mort_df
        self.years = years
        self.tensor_storage = list(range(len(self.years)))  # One data vector for each year

    def __len__(self):
        return len(self.tensor_storage)
    
    def __getitem__(self, idx):
        year = self.years[idx]
        mort_rates = self.mort_df[f'{year} Mortality Rates'].values
        mort_rates_array = np.array(mort_rates)
        mort_rates_tensor = torch.tensor(mort_rates_array, dtype=torch.float32)
        return mort_rates_tensor, mort_rates_tensor
    
class Autoencoder_model(nn.Module):
    def __init__(self):
        super(Autoencoder_model, self).__init__()                   

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NUM_COUNTIES, 2000),
                nn.ReLU(),
                nn.Linear(2000, 1000) )
        ])
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1000, 2000),
                nn.ReLU(),
                nn.Linear(2000, NUM_COUNTIES + 3) )
        ])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = x.squeeze(0)  # Remove the batch dimension
            x = layer(x)
        return x

def train_model(train_loader, model, loss_function, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    for epoch in range(num_epochs):  # Epoch loop
        model.train()  # Set the model to training mode
        epoch_loss = 0.0  # Reset epoch loss
        num_batches = 0  # Batch counter

        for inputs, targets in train_loader:  # For each batch
            optimizer.zero_grad()  # Reset gradients
            inputs = inputs.squeeze(dim=0)  # Remove the batch dimension
            targets = targets.squeeze(dim=0)  # Remove the batch dimension
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            scheduler.step()  # Update the scheduler at the end of each batch
            epoch_loss += loss.item()  # Accumulate loss
            num_batches += 1  # Increment batch counter

        average_epoch_loss = round(epoch_loss / num_batches, 4)  # Compute average epoch loss
        logging.info(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {average_epoch_loss}')  # Log epoch loss

def predict_mortality_rates(predictions_loader):
    model = Autoencoder_model()
    model.load_state_dict(torch.load('New Experiment/recon_auto.pth'))
    model.eval()
    
    # Initialize an empty DataFrame with the desired column names
    years = [f'{year+1} AE Preds' for year in DATA_YEARS]
    predictions_df = pd.DataFrame(columns=years)
    year_counter = 2010
    
    with torch.no_grad():
        for inputs, _ in predictions_loader:
            inputs = inputs.squeeze(dim=0)  # Remove the batch dimension
            year_counter += 1 # 1st input is for 2015 predictions, 2nd is for 2016 and so on
            outputs = model(inputs)
            outputs_np = outputs.numpy()  # Convert tensor to numpy array
            outputs_np = np.round(outputs_np, 2)
            predictions_df[f'{year_counter} AE Preds'] = outputs_np.flatten() # Place the column vector in the appropriate column of the DataFrame

    print(predictions_df.head())
    # Save to CSV
    predictions_df.to_csv('New Experiment/reconstructions.csv', index=False)
    print("Predictions saved to CSV --------------------------------")

def main():
    data_df = construct_mort_df(MORTALITY_PATH, MORTALITY_NAMES)  # Assuming data_df is the same as mort_df
    mort_df = data_df  # No need for a separate construct function
    tensors = Tensors(data_df, mort_df, years=DATA_YEARS)

    # Use the full dataset
    full_loader = DataLoader(tensors, batch_size=1, shuffle=False, num_workers=0)

    # Train the model
    logging.info("Training model on the full dataset --------------------------------\n")
    model = Autoencoder_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # Initial LR
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=.001, step_size_up=10, mode='triangular2')
    train_model(full_loader, model, LOSS_FUNCTION, optimizer, scheduler)
    torch.save(model.state_dict(), 'New Experiment/recon_auto.pth')
    logging.info("Model training complete and saved --------------------------------\n")

    predictions_loader = DataLoader(tensors, batch_size=1, shuffle=False, num_workers=0)
    predict_mortality_rates(predictions_loader)

if __name__ == "__main__":
    main()