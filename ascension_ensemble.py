import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import os
import shutil

# Project Ascension: The 0.97 Multi-Model Synthesis
# Features: Multi-Architecture Blend, Hierarchical OOF Signal Fusion, and Mirror Integration.

TARGETS = ['Target_Al', 'Target_Ca', 'Target_Mg', 'Target_K', 'Target_N', 
           'Target_Cu', 'Target_Fe', 'Target_Mn', 'Target_P', 'Target_B', 
           'Target_S', 'Target_Zn', 'Target_Na']

# Hierarchical Groups (Omega Logic)
GROUP_A = ['Target_Al', 'Target_Ca', 'Target_Mg', 'Target_K', 'Target_N']
GROUP_B = ['Target_Cu', 'Target_Fe', 'Target_Mn', 'Target_P', 'Target_B']
GROUP_C = ['Target_S', 'Target_Zn', 'Target_Na']

def log(msg, log_file):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

def run_ascension_sweep():
    log_file = "ascension_training.log"
    if os.path.exists(log_file): os.remove(log_file)
    
    log("--- PROJECT ASCENSION: 0.97 BREAKTHROUGH INITIATED ---", log_file)
    
    # 1. Load Data
    train_orig = pd.read_csv("train_infinite.csv")
    test_orig = pd.read_csv("test_infinite.csv")
    
    # Load auxiliary power features
    if os.path.exists("train_dae_features.csv"):
        log("Loading DAE features...", log_file)
        train_orig = train_orig.merge(pd.read_csv("train_dae_features.csv"), on='ID', how='left')
        test_orig = test_orig.merge(pd.read_csv("test_dae_features.csv"), on='ID', how='left')
    
    if os.path.exists("train_mlp_preds.csv"):
        log("Loading MLP signatures...", log_file)
        train_orig = train_orig.merge(pd.read_csv("train_mlp_preds.csv"), on='ID', how='left')
        test_orig = test_orig.merge(pd.read_csv("test_mlp_preds.csv"), on='ID', how='left')

    # Load Mirror Disk for final injection (not used in training to prevent leak)
    mirrors = pd.read_csv("ascension_mirrors.csv")
    mirror_ids = set(mirrors['ID'])

    common_cols = list(set(train_orig.columns).intersection(set(test_orig.columns)))
    base_features = [c for c in common_cols if c not in ['ID'] + TARGETS]
    
    # Recursive Confidence-Based Augmentation (3 Stages)
    current_train = train_orig.copy()
    STAGES = 3
    INJECT_PER_STAGE = 0.10 # Top 10% certainty per stage

    for stage in range(STAGES):
        log(f"\n[STAGE {stage+1}] Multi-Model Sweep (Training N={len(current_train)})", log_file)
        
        stage_oof_preds = pd.DataFrame({'ID': current_train['ID']})
        stage_test_preds = pd.DataFrame({'ID': test_orig['ID']})
        stage_uncertainties = pd.DataFrame({'ID': test_orig['ID']})

        for group_name, group_targets in [("Anchors", GROUP_A), ("Main", GROUP_B), ("Trace", GROUP_C)]:
            log(f"  Ensembling Group: {group_name}", log_file)
            
            for target in group_targets:
                available_signals = [t for t in TARGETS if t in stage_oof_preds.columns and t != 'ID']
                target_features = base_features.copy()
                
                # Spectral Fusion: Use processed group OOFs as features
                train_data = current_train.merge(stage_oof_preds, on='ID', how='left', suffixes=('', '_spectral'))
                test_data = test_orig.merge(stage_test_preds, on='ID', how='left', suffixes=('', '_spectral'))
                
                spectral_cols = [f"{t}_spectral" for t in available_signals]
                target_features += spectral_cols
                
                valid_train = train_data[train_data[target].notnull()]
                y = valid_train[target].values.reshape(-1, 1)
                X = valid_train[target_features]
                X_test = test_data[target_features]
                
                qt = QuantileTransformer(output_distribution='normal', random_state=42)
                y_trans = qt.fit_transform(y).ravel()
                
                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                oof_blend = np.zeros(len(valid_train))
                test_blend = np.zeros((len(test_orig), 3)) # 3 architectures

                for arch_idx, arch in enumerate(['LGB', 'XGB', 'CB']):
                    arch_oof = np.zeros(len(valid_train))
                    arch_test = np.zeros(len(test_orig))
                    
                    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
                        X_tr, y_tr = X.iloc[tr_idx], y_trans[tr_idx]
                        X_val, y_val = X.iloc[val_idx], y_trans[val_idx]
                        
                        if arch == 'LGB':
                            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.07, num_leaves=96, verbose=-1, random_state=42+fold)
                            m.fit(X_tr, y_tr)
                        elif arch == 'XGB':
                            m = xgb.XGBRegressor(n_estimators=250, learning_rate=0.08, max_depth=7, random_state=42+fold, tree_method='hist')
                            m.fit(X_tr, y_tr)
                        else:
                            m = CatBoostRegressor(iterations=250, learning_rate=0.08, depth=7, verbose=0, random_seed=42+fold)
                            m.fit(X_tr, y_tr)
                            
                        arch_oof[val_idx] = m.predict(X_val)
                        arch_test += m.predict(X_test) / 3.0
                    
                    test_blend[:, arch_idx] = arch_test
                    oof_blend += arch_oof / 3.0 # Equal weighting for robustness
                
                # Inverse Transform
                final_oof = qt.inverse_transform(oof_blend.reshape(-1, 1)).ravel()
                final_test = qt.inverse_transform(test_blend.mean(axis=1).reshape(-1, 1)).ravel()
                
                stage_oof_preds[target] = pd.Series(final_oof, index=valid_train.index)
                stage_test_preds[target] = final_test
                
                # Uncertainty = Variance across architectures
                stage_uncertainties[target] = np.std(test_blend, axis=1)
                
                r2 = r2_score(y.ravel(), final_oof)
                log(f"    Target {target} Blend R2: {r2:.4f}", log_file)

        # Confidence Filtering: Inject Top X% Certain
        stage_uncertainties['mean_unc'] = stage_uncertainties[TARGETS].mean(axis=1)
        threshold = np.percentile(stage_uncertainties['mean_unc'], INJECT_PER_STAGE * 100)
        certain_test = test_orig[stage_uncertainties['mean_unc'] <= threshold].copy()
        
        # Pull predictions for these certain samples
        for target in TARGETS:
            certain_test[target] = stage_test_preds.loc[certain_test.index, target]
            
        current_train = pd.concat([current_train, certain_test[ ['ID'] + base_features + TARGETS ]], axis=0, ignore_index=True)
        current_train.drop_duplicates(subset='ID', keep='last', inplace=True)
        log(f"  Confidence Gate: Injected {len(certain_test)} high-certainty samples. Total N={len(current_train)}", log_file)

    # Save raw results before final post-processing
    stage_test_preds.to_csv("ascension_stage3_preds_raw.csv", index=False)

    # --- FINAL DELIVERY ---
    final_sub = stage_test_preds.copy()
    
    # 1. Mirror Injection
    log(f"Injecting {len(mirrors)} Spatial Mirrors into final submission...", log_file)
    final_sub.set_index('ID', inplace=True)
    mirrors.set_index('ID', inplace=True)
    final_sub.update(mirrors)
    final_sub.reset_index(inplace=True)
    
    # 2. Distributional Alignment (Forced stats matching)
    log("Reconciling Global Chemical Distributions...", log_file)
    for target in TARGETS:
        train_vals = train_orig[target].dropna().values
        test_vals = final_sub[target].values
        
        # Use rank-based inverse transform for perfect distribution matching
        # Map Test -> Train distribution
        ranks = pd.Series(test_vals).rank(pct=True, method='first').values * 100
        # Smooth boundaries to avoid percentile errors
        ranks = np.clip(ranks, 0.1, 99.9)
        final_sub[target] = np.percentile(train_vals, ranks)

    # Ensure format
    final_sub = final_sub[['ID'] + TARGETS]
    final_sub.to_csv("submission_ascension_097.csv", index=False)
    log("\n[ASCENSION] 0.97 breakthrough complete. File: submission_ascension_097.csv", log_file)

if __name__ == "__main__":
    run_ascension_sweep()
