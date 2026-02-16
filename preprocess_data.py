import pandas as pd
import numpy as np
import os

# Define path
DATA_PATH = r"c:\Users\Administrator.RAJESH\Desktop\22\New folder (6)\rhea-soil-nutrient-prediction-challenge20260210-28069-1fmhbqj"

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import NearestNeighbors

def add_spatial_knn_features(train, test):
    print("Adding spatial KNN features...")
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    # Scale coordinates
    coords_scaler = StandardScaler()
    train_coords = coords_scaler.fit_transform(train[['Latitude', 'Longitude']])
    test_coords = coords_scaler.transform(test[['Latitude', 'Longitude']])
    
    # Fit KNN on training coordinates only
    knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    knn.fit(train_coords)
    
    # Get neighbor indices once
    # For training itself, the 1st neighbor is always itself (dist 0), so we use 1:11
    # For test, we use 0:10
    _, train_indices = knn.kneighbors(train_coords, n_neighbors=11)
    _, test_indices = knn.kneighbors(test_coords, n_neighbors=10)
    
    # For each target, get the mean of neighboring values
    for target in targets:
        # Mean of neighbors (excluding self for train)
        train_target_vals = train[target].values
        train[f'knn_mean_{target}'] = train_target_vals[train_indices[:, 1:]].mean(axis=1)
        test[f'knn_mean_{target}'] = train_target_vals[test_indices[:, :10]].mean(axis=1)
        
        # Fill NaNs in KNN features (if any) with training mean
        t_mean = train[target].mean()
        train[f'knn_mean_{target}'].fillna(t_mean, inplace=True)
        test[f'knn_mean_{target}'].fillna(t_mean, inplace=True)
        
    return train, test

def preprocess():
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_PATH, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_PATH, "TestSet.csv"))
    dates = pd.read_csv(os.path.join(DATA_PATH, "Sample_Collection_Dates.csv"))
    
    # Merge dates if not already present
    if 'start_date' not in test.columns:
        print("Merging dates with Test set...")
        test = test.merge(dates[['ID', 'start_date', 'end_date']], on='ID', how='left')
    
    # Mapping Train targets to match Test target names (Al, B, etc)
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    column_mapping = {t.split('_')[1]: t for t in targets}
    train.rename(columns=column_mapping, inplace=True)

    # Spatial KNN Features
    train, test = add_spatial_knn_features(train, test)

    # Convert Depth_cm range to numeric midpoint
    def convert_depth(depth_str):
        if not isinstance(depth_str, str):
            return depth_str
        try:
            low, high = map(float, depth_str.split('-'))
            return (low + high) / 2.0
        except:
            return depth_str

    # Spatial Clustering with Scaling
    print("Performing spatial clustering with scaling...")
    coords = pd.concat([train[['Latitude', 'Longitude']], test[['Latitude', 'Longitude']]], axis=0)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
    coords['cluster'] = kmeans.fit_predict(coords_scaled)
    
    train['cluster'] = coords['cluster'].iloc[:len(train)].values
    test['cluster'] = coords['cluster'].iloc[len(train):].values

    for df in [train, test]:
        df['Depth_cm'] = df['Depth_cm'].apply(convert_depth)
        
        # Spatial Interactions
        df['lat_lon'] = df['Latitude'] * df['Longitude']
        df['lat_sq'] = df['Latitude'] ** 2
        df['lon_sq'] = df['Longitude'] ** 2
        df['lat_depth'] = df['Latitude'] * df['Depth_cm']
        df['lon_depth'] = df['Longitude'] * df['Depth_cm']
        df['rot_45_x'] = 0.707 * df['Latitude'] + 0.707 * df['Longitude']
        df['rot_45_y'] = 0.707 * df['Longitude'] - 0.707 * df['Latitude']

        df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')
        df['start_month'] = df['start_date'].dt.month
        df['start_dayofyear'] = df['start_date'].dt.dayofyear
        
        df['month_sin'] = np.sin(2 * np.pi * df['start_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['start_month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['start_dayofyear'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['start_dayofyear'] / 365)
        df['year'] = df['start_date'].dt.year
        
        df.drop(['start_date', 'end_date', 'start_month', 'start_dayofyear'], axis=1, inplace=True)

    # Save preprocessed data
    train.to_csv("train_processed.csv", index=False)
    test.to_csv("test_processed.csv", index=False)
    print("Preprocessing complete with KNN Spatial features.")

if __name__ == "__main__":
    preprocess()
