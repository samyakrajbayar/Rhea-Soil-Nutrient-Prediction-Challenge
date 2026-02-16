import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def add_infinite_features(train, test):
    print("Applying Infinite Spatial Refinery...")
    
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    # 1. Rotated Coordinates (Additional Perspectives)
    for df in [train, test]:
        for angle in [30, 45, 60]:
            rad = np.radians(angle)
            df[f'rot_{angle}_x'] = df['Latitude'] * np.cos(rad) + df['Longitude'] * np.sin(rad)
            df[f'rot_{angle}_y'] = df['Longitude'] * np.cos(rad) - df['Latitude'] * np.sin(rad)

    # 2. Inverse Distance Weighting (IDW) & Neighborhood Stats
    scaler = StandardScaler()
    train_coords = scaler.fit_transform(train[['Latitude', 'Longitude']])
    test_coords = scaler.transform(test[['Latitude', 'Longitude']])
    
    n_neighbors = 10
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(train_coords)
    
    # Neighbors for training (excluding self)
    dist_tr, idx_tr = knn.kneighbors(train_coords, n_neighbors=n_neighbors + 1)
    dist_te, idx_te = knn.kneighbors(test_coords, n_neighbors=n_neighbors)
    
    for target in targets:
        print(f"Processing neighbor features for {target}...")
        train_vals = train[target].values
        
        # IDW Calculation: weighting by 1/dist
        # Training
        epsilon = 1e-5
        weights_tr = 1.0 / (dist_tr[:, 1:] + epsilon)
        neighbor_vals_tr = train_vals[idx_tr[:, 1:]]
        # Handle cases where training values might be NaN (though target shouldn't be for its own model)
        # But we provide these features to ALL models, so we must handle NaNs
        
        # Simple IDW mean
        train[f'idw_{target}'] = np.sum(neighbor_vals_tr * weights_tr, axis=1) / np.sum(weights_tr, axis=1)
        
        # Stats
        train[f'neigh_mean_{target}'] = np.nanmean(neighbor_vals_tr, axis=1)
        train[f'neigh_std_{target}'] = np.nanstd(neighbor_vals_tr, axis=1)
        train[f'neigh_max_{target}'] = np.nanmax(neighbor_vals_tr, axis=1)
        train[f'neigh_min_{target}'] = np.nanmin(neighbor_vals_tr, axis=1)
        
        # Test
        weights_te = 1.0 / (dist_te + epsilon)
        neighbor_vals_te = train_vals[idx_te]
        
        test[f'idw_{target}'] = np.sum(neighbor_vals_te * weights_te, axis=1) / np.sum(weights_te, axis=1)
        test[f'neigh_mean_{target}'] = np.nanmean(neighbor_vals_te, axis=1)
        test[f'neigh_std_{target}'] = np.nanstd(neighbor_vals_te, axis=1)
        test[f'neigh_max_{target}'] = np.nanmax(neighbor_vals_te, axis=1)
        test[f'neigh_min_{target}'] = np.nanmin(neighbor_vals_te, axis=1)
        
        # Global NaNs fill
        t_mean = np.nanmean(train_vals)
        for col in [f'idw_{target}', f'neigh_mean_{target}', f'neigh_std_{target}', f'neigh_max_{target}', f'neigh_min_{target}']:
            train[col] = train[col].fillna(t_mean)
            test[col] = test[col].fillna(t_mean)
            
    return train, test

def preprocess_infinite():
    print("Loading core data...")
    # Loading base processed data
    train = pd.read_csv("train_processed.csv")
    test = pd.read_csv("test_processed.csv")
    
    # Apply Project Infinite Refinery
    train, test = add_infinite_features(train, test)
    
    # Save Infinite files
    train.to_csv("train_infinite.csv", index=False)
    test.to_csv("test_infinite.csv", index=False)
    print("Preprocessing Infinite complete.")

if __name__ == "__main__":
    preprocess_infinite()
