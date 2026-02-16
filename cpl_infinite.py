import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
import os

def generate_cpl():
    print("Initiating Project Infinite CPL Refinery...")
    
    # 1. Load Data
    train = pd.read_csv("train_infinite.csv")
    test = pd.read_csv("test_infinite.csv")
    submission_baseline = pd.read_csv("submission_infinite.csv")
    
    if os.path.exists("train_dae_features.csv"):
        print("Merging DAE features...")
        train = train.merge(pd.read_csv("train_dae_features.csv"), on='ID')
        test = test.merge(pd.read_csv("test_dae_features.csv"), on='ID')
        
    if os.path.exists("train_mlp_preds.csv"):
        print("Merging MLP signatures...")
        train = train.merge(pd.read_csv("train_mlp_preds.csv"), on='ID')
        test = test.merge(pd.read_csv("test_mlp_preds.csv"), on='ID')

    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    common_cols = list(set(train.columns).intersection(set(test.columns)))
    features = [c for c in common_cols if c not in ['ID'] + targets]
    
    print(f"Refining across {len(features)} manifold dimensions.")
    
    pseudo_samples = []
    
    for target in targets:
        print(f"\nEvaluating Confidence for {target}...")
        
        valid_idx = train[train[target].notnull()].index
        X = train.loc[valid_idx, features]
        y = train.loc[valid_idx, target].values.reshape(-1, 1)
        
        # Transform for variance stability
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        y_trans = qt.fit_transform(y).ravel()
        
        # 3-Seed Uncertainty Check
        seeds = [1, 2, 3]
        test_preds_all = np.zeros((len(test), len(seeds)))
        
        for s_idx, seed in enumerate(seeds):
            # Fast LGBM for uncertainty proxy
            m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.1, random_state=seed, verbose=-1)
            m.fit(X, y_trans)
            test_preds_all[:, s_idx] = m.predict(test[features])
            
        # Calculate Coefficient of Variation (Std/Mean approx in transformed space)
        # We use simple variance as a proxy for 'certainty' in high-dimensional reg
        uncertainty = np.std(test_preds_all, axis=1)
        
        # Select Top 20% most certain samples (lowest variance)
        threshold = np.percentile(uncertainty, 20)
        certain_idx = np.where(uncertainty <= threshold)[0]
        
        print(f"  Identified {len(certain_idx)} high-confidence test samples for {target}.")
        
        # Create pseudo-data
        target_pseudo = test.iloc[certain_idx][['ID'] + features].copy()
        target_pseudo[target] = submission_baseline.iloc[certain_idx][target].values
        
        # Keep only target-specific pseudo labels to prevent cross-contamination of uncertain labels
        # (The rest of the targets for these IDs will be NaN in this subset)
        for t in targets:
            if t != target:
                target_pseudo[t] = np.nan
                
        pseudo_samples.append(target_pseudo)
        
    # 2. Combine and Augment
    all_pseudo = pd.concat(pseudo_samples, ignore_index=True)
    
    # Merge targets by ID (if an ID is certain for multiple targets)
    all_pseudo = all_pseudo.groupby('ID', as_index=False).first()
    
    # Final Augmentation
    train_cpl = pd.concat([train, all_pseudo], ignore_index=True)
    
    print(f"\nCPL complete. Training set expanded from {len(train)} to {len(train_cpl)} samples.")
    train_cpl.to_csv("train_infinite_cpl.csv", index=False)
    print("Saved train_infinite_cpl.csv")

if __name__ == "__main__":
    generate_cpl()
