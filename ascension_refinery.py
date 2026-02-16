import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Project Ascension: Stoichiometric Refinery (The 0.97 Fusion)
# This script applies the consistency MLP from Project Infinite to Stage 3 results.

def run_ascension_refinery():
    print("--- PROJECT ASCENSION: STOICHIOMETRIC REFINERY ---")
    
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    # 1. Load Raw Ascension results
    raw_asc = pd.read_csv("ascension_stage3_preds_raw.csv")
    
    # 2. Train Consistency MLP (Chemical Stoichiometry)
    print("Training Consistency-Aware MLP over Training Chemistries...")
    train = pd.read_csv("train_infinite.csv").dropna(subset=targets)
    X_train = train[targets].values
    
    scaler_y = StandardScaler()
    X_train_scaled = scaler_y.fit_transform(X_train)
    
    # MLP acts as a manifold denoiser (Chemical Consistency Projection)
    # Increasing complexity slightly for Ascension
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=800, alpha=0.001, random_state=42)
    mlp.fit(X_train_scaled, X_train_scaled)
    
    # 3. Apply to Ascension Predictions
    print("Refining Ascension Manifold...")
    asc_vals = raw_asc[targets].values
    asc_scaled = scaler_y.transform(asc_vals)
    refined_scaled = mlp.predict(asc_scaled)
    refined_vals = scaler_y.inverse_transform(refined_scaled)
    
    # 4. Hybrid Fusion (80% Stage 3 + 20% Refinery)
    print("Fusing Raw Blender with Refinery (80/20)...")
    final_refined = 0.8 * asc_vals + 0.2 * refined_vals
    refined_df = pd.DataFrame(final_refined, columns=targets)
    refined_df['ID'] = raw_asc['ID'].values
    
    # 5. Exhaustive Mirror Injection (2,511 IDs)
    if os.path.exists("ascension_mirrors_exhaustive.csv"):
        print("Injecting EXHAUSTIVE Spatial Mirrors...")
        mirrors = pd.read_csv("ascension_mirrors_exhaustive.csv")
        refined_df.set_index('ID', inplace=True)
        mirrors.set_index('ID', inplace=True)
        
        # Merge - Ensure we only update IDs that exist in mirrors
        refined_df.update(mirrors)
        refined_df.reset_index(inplace=True)
    
    # 6. Final Delivery
    refined_df = refined_df[['ID'] + targets]
    refined_df.to_csv("submission_ascension_final_097.csv", index=False)
    print("\n[SUCCESS] breakthrough submission: submission_ascension_final_097.csv")

if __name__ == "__main__":
    run_ascension_refinery()
