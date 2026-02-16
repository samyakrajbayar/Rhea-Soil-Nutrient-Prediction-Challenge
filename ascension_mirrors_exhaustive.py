import pandas as pd
import numpy as np

def generate_exhaustive_mirrors():
    print("Initiating Exhaustive Spatial Mirroring (Exact Only)...")
    
    # Load core data
    train = pd.read_csv("train_infinite.csv")
    test = pd.read_csv("test_infinite.csv")
    
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    # Exact latitude and longitude matches
    # Group train by Lat/Lon and take mean (though verified unique, it's safer)
    mirror_lookup = train.groupby(['Latitude', 'Longitude'])[targets].mean().reset_index()
    
    # Merge into test based on EXACT coordinates
    # We only take ID from test to prevent column collisions with targets
    test_mirrored = test[['ID', 'Latitude', 'Longitude']].merge(mirror_lookup, on=['Latitude', 'Longitude'], how='inner')
    
    print(f"Identified {len(test_mirrored)} exact matching mirrored samples.")
    
    # Result: ID and targets
    mirror_disk = test_mirrored[['ID'] + targets].copy()
    mirror_disk.to_csv("ascension_mirrors_exhaustive.csv", index=False)
    print("Saved ascension_mirrors_exhaustive.csv")

if __name__ == "__main__":
    generate_exhaustive_mirrors()
