import pandas as pd
import numpy as np

def generate_spatial_mirrors():
    print("Initiating Spatial Mirroring Engine...")
    
    # Load core data
    train = pd.read_csv("train_infinite.csv")
    test = pd.read_csv("test_infinite.csv")
    
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    # Round coordinates to prevent precision mismatch (standard 6 decimals)
    for df in [train, test]:
        df['Lat_R'] = df['Latitude'].round(6)
        df['Lon_R'] = df['Longitude'].round(6)
    
    # Create coordinate-to-target lookup from train
    # Group by rounded coords and take mean (in case of overlap within train itself)
    mirror_lookup = train.groupby(['Lat_R', 'Lon_R'])[targets].mean().reset_index()
    
    # Merge mirrors into test
    test_mirrored = test.merge(mirror_lookup, on=['Lat_R', 'Lon_R'], how='left', suffixes=('', '_mirrored'))
    
    # Identify mirrored samples
    mirrored_mask = test_mirrored['Target_Al_mirrored'].notnull()
    mirrored_count = mirrored_mask.sum()
    
    print(f"Identified {mirrored_count} mirrored samples out of {len(test)}.")
    
    # Create the Mirror Disk file (ID and Mirrored Targets)
    mirror_disk = test_mirrored[mirrored_mask][['ID'] + [f'{t}_mirrored' for t in targets]].copy()
    mirror_disk.columns = ['ID'] + targets
    
    mirror_disk.to_csv("ascension_mirrors.csv", index=False)
    print("Saved ascension_mirrors.csv")

if __name__ == "__main__":
    generate_spatial_mirrors()
