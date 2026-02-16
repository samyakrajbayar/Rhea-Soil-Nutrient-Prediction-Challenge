import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import os

def train_mlp():
    print("Loading data for Multi-Target MLP...")
    train_proc = pd.read_csv("train_processed.csv")
    test_proc = pd.read_csv("test_processed.csv")
    train_dae = pd.read_csv("train_dae_features.csv")
    test_dae = pd.read_csv("test_dae_features.csv")
    
    # Merge DAE features
    train = train_proc.merge(train_dae, on='ID')
    test = test_proc.merge(test_dae, on='ID')
    
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    # Common features for MLP (Intersection of common columns but excluding targets)
    common_cols = list(set(train_proc.columns).intersection(set(test_proc.columns)))
    features_base = [c for c in common_cols if c not in ['ID'] + targets]
    
    # Add DAE latent features
    dae_features = [c for c in train_dae.columns if c != 'ID']
    features = features_base + dae_features
    
    print(f"Using {len(features)} total features for MLP.")
    X = train[features].values.astype(float)
    X_test_all = test[features].values.astype(float)
    y = train[targets].values.astype(float)
    
    # Scale X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.vstack([X, X_test_all]))
    X = X_scaled[:len(X)]
    X_test_all = X_scaled[len(X):]
    
    # Transform y (handle NaNs per target)
    y_trans = np.zeros_like(y)
    transformers = {}
    for i, target in enumerate(targets):
        valid_idx = ~np.isnan(y[:, i])
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        y_trans[valid_idx, i] = qt.fit_transform(y[valid_idx, i].reshape(-1, 1)).ravel()
        transformers[target] = qt
        
    class MultiTargetMLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(MultiTargetMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
            
        def forward(self, x):
            return self.net(x)

    def masked_mse_loss(y_pred, y_true):
        mask = ~torch.isnan(y_true)
        diff = (y_pred[mask] - y_true[mask])**2
        return diff.mean()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros_like(y_trans)
    test_preds = np.zeros((len(test), len(targets)))
    
    print("Starting 5-fold CV for MLP...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = torch.FloatTensor(X[train_idx]), torch.FloatTensor(y_trans[train_idx])
        X_val, y_val = torch.FloatTensor(X[val_idx]), torch.FloatTensor(y_trans[val_idx])
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
        
        model = MultiTargetMLP(X.shape[1], len(targets))
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(100):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = masked_mse_loss(pred, by)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = masked_mse_loss(val_pred, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    oof_preds[val_idx] = val_pred.numpy()
                    # Best model for test inference
                    current_test_pred = model(torch.FloatTensor(X_test_all)).numpy()
                
            scheduler.step(val_loss)
            if (epoch + 1) % 20 == 0:
                print(f"Fold {fold+1}, Epoch {epoch+1}, Val Loss: {val_loss.item():.6f}")
                
        test_preds += current_test_pred / 5.0
        print(f"Fold {fold+1} complete. Best Val Loss: {best_val_loss:.6f}")

    # Back-transform and evaluate
    final_oof = np.zeros_like(y)
    final_test = np.zeros((len(test), len(targets)))
    
    overall_r2 = []
    for i, target in enumerate(targets):
        qt = transformers[target]
        valid_idx = ~np.isnan(y[:, i])
        
        final_oof[valid_idx, i] = qt.inverse_transform(oof_preds[valid_idx, i].reshape(-1, 1)).ravel()
        final_test[:, i] = qt.inverse_transform(test_preds[:, i].reshape(-1, 1)).ravel()
        
        r2 = r2_score(y[valid_idx, i], final_oof[valid_idx, i])
        overall_r2.append(r2)
        print(f"Target {target} MLP R2: {r2:.4f}")
        
    print(f"Overall MLP Mean R2: {np.mean(overall_r2):.4f}")
    
    # Save MLP OOF and test predictions for stacking level 2
    oof_df = pd.DataFrame(final_oof, columns=[f'mlp_{t}' for t in targets])
    oof_df['ID'] = train['ID'].values
    test_df = pd.DataFrame(final_test, columns=[f'mlp_{t}' for t in targets])
    test_df['ID'] = test['ID'].values
    
    oof_df.to_csv("train_mlp_preds.csv", index=False)
    test_df.to_csv("test_mlp_preds.csv", index=False)
    print("MLP predictions saved.")

if __name__ == "__main__":
    train_mlp()
