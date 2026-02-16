import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import os

def train_infinite():
    log_file = open("infinite_training.log", "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("Loading CPL Augmented Infinite data...")
    train = pd.read_csv("train_infinite_cpl.csv")
    test = pd.read_csv("test_infinite.csv")
    
    # Load DAE and MLP features if they exist for extra boost
    if os.path.exists("train_dae_features.csv"):
        log("Merging DAE features...")
        train_dae = pd.read_csv("train_dae_features.csv")
        test_dae = pd.read_csv("test_dae_features.csv")
        # Ensure we only merge for rows present in CPL train
        train = train.merge(train_dae, on='ID', how='left')
        test = test.merge(test_dae, on='ID', how='left')
        
    if os.path.exists("train_mlp_preds.csv"):
        log("Merging MLP signatures...")
        train_mlp = pd.read_csv("train_mlp_preds.csv")
        test_mlp = pd.read_csv("test_mlp_preds.csv")
        train = train.merge(train_mlp, on='ID', how='left')
        test = test.merge(test_mlp, on='ID', how='left')

    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    # Common features
    common_cols = list(set(train.columns).intersection(set(test.columns)))
    features = [c for c in common_cols if c not in ['ID'] + targets]
    
    log(f"Using {len(features)} total features for Refined CPL Ensemble.")
    
    submission = pd.DataFrame(index=test.index)
    submission['ID'] = test['ID']
    overall_r2 = []
    
    for target in targets:
        log(f"\n--- Refined CPL Ensemble for {target} ---")
        
        # Train only on rows where the target (including pseudo-labels) is available
        valid_idx = train[train[target].notnull()].index
        X = train.loc[valid_idx, features]
        y = train.loc[valid_idx, target].values.reshape(-1, 1)
        
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        y_trans = qt.fit_transform(y).ravel()
        
        # 3-Fold CV, 3 Seeds, 3 Architectures = 27 models per target
        seeds = [42, 2024, 888]
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        oof_lgb = np.zeros((len(X), len(seeds)))
        oof_xgb = np.zeros((len(X), len(seeds)))
        oof_cb = np.zeros((len(X), len(seeds)))
        
        test_lgb = np.zeros((len(test), len(seeds)))
        test_xgb = np.zeros((len(test), len(seeds)))
        test_cb = np.zeros((len(test), len(seeds)))
        
        for s_idx, seed in enumerate(seeds):
            log(f"  Refining Seed {seed}...")
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
                X_tr, y_tr =  X.iloc[tr_idx], y_trans[tr_idx]
                X_val, y_val = X.iloc[val_idx], y_trans[val_idx]
                
                # Boosted hyperparams for refinement
                m_lgb = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=127, 
                                          feature_fraction=0.7, bagging_fraction=0.7, random_state=seed, verbose=-1)
                m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(40)])
                oof_lgb[val_idx, s_idx] = m_lgb.predict(X_val)
                test_lgb[:, s_idx] += m_lgb.predict(test[features]) / 3.0
                
                m_xgb = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8, 
                                          subsample=0.7, colsample_bytree=0.7, random_state=seed, tree_method='hist')
                m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                oof_xgb[val_idx, s_idx] = m_xgb.predict(X_val)
                test_xgb[:, s_idx] += m_xgb.predict(test[features]) / 3.0
                
                cb_dir = f"catboost_info_refined_{target}_{seed}_{fold}"
                m_cb = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=8, 
                                          random_seed=seed, verbose=0, early_stopping_rounds=40,
                                          train_dir=cb_dir)
                m_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
                oof_cb[val_idx, s_idx] = m_cb.predict(X_val)
                test_cb[:, s_idx] += m_cb.predict(test[features]) / 3.0
                
        X_blend = np.column_stack([oof_lgb.mean(axis=1), oof_xgb.mean(axis=1), oof_cb.mean(axis=1)])
        X_test_blend = np.column_stack([test_lgb.mean(axis=1), test_xgb.mean(axis=1), test_cb.mean(axis=1)])
        
        blender = Ridge(alpha=0.5) # Dynamic alpha for tighter manifold
        blender.fit(X_blend, y_trans)
        
        final_oof_trans = blender.predict(X_blend)
        final_test_trans = blender.predict(X_test_blend)
        
        final_oof = qt.inverse_transform(final_oof_trans.reshape(-1, 1)).ravel()
        final_test = qt.inverse_transform(final_test_trans.reshape(-1, 1)).ravel()
        
        r2 = r2_score(y.ravel(), final_oof)
        log(f"Target {target} Refined R2: {r2:.4f}")
        overall_r2.append(r2)
        
        submission[target] = final_test
        
    log(f"\n[Project Infinite] FINAL Refined Mean R2: {np.mean(overall_r2):.4f}")
    submission.to_csv("submission_infinite_refined.csv", index=False)
    log_file.close()

if __name__ == "__main__":
    train_infinite()
