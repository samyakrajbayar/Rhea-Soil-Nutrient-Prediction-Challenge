import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def train_dae():
    print("Loading data for DAE...")
    train = pd.read_csv("train_processed.csv")
    test = pd.read_csv("test_processed.csv")
    
    # Select features common to both train and test
    common_cols = list(set(train.columns).intersection(set(test.columns)))
    features = [c for c in common_cols if c not in ['ID'] and not c.startswith('Target_')]
    
    print(f"Using {len(features)} common features for DAE.")
    X_train = train[features].values.astype(float)
    X_test = test[features].values.astype(float)
    X_all = np.vstack([X_train, X_test])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    # Convert to torch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    class DAE(nn.Module):
        def __init__(self, input_dim):
            super(DAE, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64) # Latent bottleneck
            )
            self.decoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim)
            )
            
        def forward(self, x):
            # Add Gaussian noise
            noise = torch.randn_like(x) * 0.1
            encoded = self.encoder(x + noise)
            decoded = self.decoder(encoded)
            return decoded

    input_dim = X_scaled.shape[1]
    model = DAE(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("Training DAE...")
    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")
            
    # Extract latent features
    model.eval()
    with torch.no_grad():
        X_latent = model.encoder(X_tensor).numpy()
        
    # Standardize latent features
    latent_scaler = StandardScaler()
    X_latent = latent_scaler.fit_transform(X_latent)
    
    # Create column names
    latent_cols = [f'dae_{i}' for i in range(X_latent.shape[1])]
    latent_df = pd.DataFrame(X_latent, columns=latent_cols)
    
    # Split back to train and test
    train_latent = latent_df.iloc[:len(train)].copy()
    test_latent = latent_df.iloc[len(train):].copy()
    
    # Save latent features
    train_latent['ID'] = train['ID'].values
    test_latent['ID'] = test['ID'].values
    
    train_latent.to_csv("train_dae_features.csv", index=False)
    test_latent.to_csv("test_dae_features.csv", index=False)
    print("DAE feature extraction complete.")

if __name__ == "__main__":
    train_dae()
