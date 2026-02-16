import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Define path
DATA_PATH = r"c:\Users\Administrator.RAJESH\Desktop\22\New folder (6)\rhea-soil-nutrient-prediction-challenge20260210-28069-1fmhbqj"

def add_spatial_magic(train, test):
    print("Applying Spatial Magic...")
    
    # 1. Coordinate Binning (Captures regional rectangular boxes)
    for df in [train, test]:
        df['lat_bin_2'] = df['Latitude'].round(2)
        df['lon_bin_2'] = df['Longitude'].round(2)
        df['lat_bin_3'] = df['Latitude'].round(3)
        df['lon_bin_3'] = df['Longitude'].round(3)
        
    # 2. Distance to Cluster Centroids
    coords = pd.concat([train[['Latitude', 'Longitude']], test[['Latitude', 'Longitude']]], axis=0)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(coords_scaled)
    
    # Calculate distance to each of the 20 centroids
    dists = kmeans.transform(coords_scaled)
    dist_cols = [f'dist_centroid_{i}' for i in range(20)]
    dist_df = pd.DataFrame(dists, columns=dist_cols, index=coords.index)
    
    train = pd.concat([train, dist_df.iloc[:len(train)]], axis=1)
    test = pd.concat([test, dist_df.iloc[len(train):]], axis=1)
    
    # 3. Target Nearest Neighbor (TNN) - Raw Neighbor Values
    # Use 3-nearest training neighbors
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    train_coords = scaler.transform(train[['Latitude', 'Longitude']])
    test_coords = scaler.transform(test[['Latitude', 'Longitude']])
    
    knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    knn.fit(train_coords)
    
    _, train_indices = knn.kneighbors(train_coords, n_neighbors=4) # 1st is self
    _, test_indices = knn.kneighbors(test_coords, n_neighbors=3)
    
    for target in targets:
        vals = train[target].values
        # Take the raw value of the 1st nearest neighbor (the most similar spot)
        train[f'tnn_1_{target}'] = vals[train_indices[:, 1]]
        train[f'tnn_2_{target}'] = vals[train_indices[:, 2]]
        
        test[f'tnn_1_{target}'] = vals[test_indices[:, 0]]
        test[f'tnn_2_{target}'] = vals[test_indices[:, 1]]
        
    return train, test

def preprocess_omega():
    print("Loading core data...")
    # Loading already processed data to build upon it
    train = pd.read_csv("train_processed.csv")
    test = pd.read_csv("test_processed.csv")
    
    # Apply Omega Magic
    train, test = add_spatial_magic(train, test)
    
    # Save Omega files
    train.to_csv("train_omega.csv", index=False)
    test.to_csv("test_omega.csv", index=False)
    print("Preprocessing Omega complete.")

if __name__ == "__main__":
    preprocess_omega()
