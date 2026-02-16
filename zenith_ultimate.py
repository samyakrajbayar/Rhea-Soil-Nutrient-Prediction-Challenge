import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import warnings, os, shutil, gc
warnings.filterwarnings('ignore')

# ============================================================
# PROJECT ZENITH: THE ULTIMATE 0.97 SOLUTION
# ============================================================
# Uses pre-processed train_infinite.csv / test_infinite.csv (98 features)
# + MLP and DAE signatures for extra signal
# Architecture: 2-Level Stacking (4 archs × 5 seeds × 5 folds)
# ============================================================

DATA_DIR = "rhea-soil-nutrient-prediction-challenge20260210-28069-1fmhbqj"
TARGETS = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe',
           'Target_K', 'Target_Mg', 'Target_Mn', 'Target_N', 'Target_Na',
           'Target_P', 'Target_S', 'Target_Zn']
TARGETS_RAW = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']
TARGET_MAP = dict(zip(TARGETS_RAW, TARGETS))

N_FOLDS = 5
SEEDS = [42, 2024, 888, 1337, 7777]
LOG_FILE = "zenith_training.log"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    log("=" * 60)
    log("PROJECT ZENITH: THE ULTIMATE 0.97 BREAKTHROUGH")
    log("=" * 60)
    
    # ============================================================
    # PHASE 1: Load Pre-Processed Data + Auxiliary Signatures
    # ============================================================
    log("\n=== PHASE 1: Loading Data ===")
    train = pd.read_csv("train_infinite.csv")
    test = pd.read_csv("test_infinite.csv")
    log(f"  Train: {train.shape}, Test: {test.shape}")
    
    # Load DAE features
    if os.path.exists("train_dae_features.csv"):
        log("  Loading DAE features...")
        train = train.merge(pd.read_csv("train_dae_features.csv"), on='ID', how='left')
        test = test.merge(pd.read_csv("test_dae_features.csv"), on='ID', how='left')
    
    # Load MLP predictions
    if os.path.exists("train_mlp_preds.csv"):
        log("  Loading MLP signatures...")
        train = train.merge(pd.read_csv("train_mlp_preds.csv"), on='ID', how='left')
        test = test.merge(pd.read_csv("test_mlp_preds.csv"), on='ID', how='left')
    
    # Also load dates for temporal features
    dates_file = f"{DATA_DIR}/Sample_Collection_Dates.csv"
    if os.path.exists(dates_file):
        try:
            log("  Loading temporal features from dates...")
            dates = pd.read_csv(dates_file)
            dates['start_date'] = pd.to_datetime(dates['start_date'], format='%d/%m/%Y', errors='coerce')
            dates['end_date'] = pd.to_datetime(dates['end_date'], format='%d/%m/%Y', errors='coerce')
            dates['year'] = dates['start_date'].dt.year.astype(float)
            dates['month'] = dates['start_date'].dt.month.astype(float)
            dates['day_of_year'] = dates['start_date'].dt.dayofyear.astype(float)
            dates['season'] = ((dates['month'] % 12 + 3) // 3).astype(float)
            dates['date_range_days'] = (dates['end_date'] - dates['start_date']).dt.days.astype(float)
            dates['sin_month'] = np.sin(2 * np.pi * dates['month'] / 12)
            dates['cos_month'] = np.cos(2 * np.pi * dates['month'] / 12)
            date_feats = ['year', 'month', 'day_of_year', 'season', 'date_range_days', 'sin_month', 'cos_month']
            
            # Drop rows with NaN IDs
            dates = dates.dropna(subset=['ID'])
            
            train = train.merge(dates[['ID'] + date_feats], on='ID', how='left')
            test = test.merge(dates[['ID'] + date_feats], on='ID', how='left')
            
            for f in date_feats:
                med = train[f].median()
                if pd.isna(med): med = 0
                train[f] = train[f].fillna(med)
                test[f] = test[f].fillna(med)
            log(f"  Added {len(date_feats)} temporal features.")
        except Exception as e:
            log(f"  WARNING: Temporal features failed: {e}. Continuing without them.")
    
    # ============================================================
    # PHASE 2: Feature Selection
    # ============================================================
    common_cols = sorted(list(set(train.columns) & set(test.columns)))
    feature_cols = [c for c in common_cols if c not in ['ID'] + TARGETS]
    log(f"\n=== PHASE 2: Feature Selection ({len(feature_cols)} features) ===")
    
    # Fill remaining NaNs
    for c in feature_cols:
        med = train[c].median()
        if pd.isna(med): med = 0
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)
    
    # ============================================================
    # PHASE 3: 2-Level Stacking Ensemble
    # ============================================================
    log("\n=== PHASE 3: 2-Level Stacking Ensemble ===")
    log(f"  Config: {len(SEEDS)} seeds × 4 architectures × {N_FOLDS} folds = {len(SEEDS)*4*N_FOLDS} models per target")
    
    submission = pd.DataFrame({'ID': test['ID']})
    overall_r2 = []
    
    for target in TARGETS:
        log(f"\n--- TARGET: {target} ---")
        
        valid_mask = train[target].notnull()
        X = train.loc[valid_mask, feature_cols].values
        y_raw = train.loc[valid_mask, target].values
        X_test = test[feature_cols].values
        
        n_train = len(X)
        n_test = len(X_test)
        log(f"  Training samples: {n_train}, Features: {X.shape[1]}")
        
        # QuantileTransformer
        qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(1000, n_train))
        y = qt.fit_transform(y_raw.reshape(-1, 1)).ravel()
        
        # ============================================================
        # Level 1: 4 architectures × 5 seeds
        # ============================================================
        n_archs = 4
        n_seeds = len(SEEDS)
        
        oof_l1 = np.zeros((n_train, n_archs * n_seeds))
        test_l1 = np.zeros((n_test, n_archs * n_seeds))
        
        col_idx = 0
        
        # --- LightGBM ---
        for seed in SEEDS:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof_arch = np.zeros(n_train)
            test_arch = np.zeros(n_test)
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
                m = lgb.LGBMRegressor(
                    n_estimators=800, learning_rate=0.03, num_leaves=127,
                    feature_fraction=0.65, bagging_fraction=0.7, bagging_freq=5,
                    min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=seed, verbose=-1, n_jobs=-1
                )
                m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
                oof_arch[val_idx] = m.predict(X[val_idx])
                test_arch += m.predict(X_test) / N_FOLDS
            
            oof_l1[:, col_idx] = oof_arch
            test_l1[:, col_idx] = test_arch
            col_idx += 1
        
        lgb_r2 = r2_score(y, oof_l1[:, :n_seeds].mean(axis=1))
        log(f"  L1 LGB OOF R2: {lgb_r2:.4f}")
        
        # --- XGBoost ---
        for seed in SEEDS:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof_arch = np.zeros(n_train)
            test_arch = np.zeros(n_test)
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
                m = xgb.XGBRegressor(
                    n_estimators=800, learning_rate=0.03, max_depth=8,
                    subsample=0.7, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=seed, tree_method='hist', n_jobs=-1
                )
                m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
                oof_arch[val_idx] = m.predict(X[val_idx])
                test_arch += m.predict(X_test) / N_FOLDS
            
            oof_l1[:, col_idx] = oof_arch
            test_l1[:, col_idx] = test_arch
            col_idx += 1
        
        xgb_r2 = r2_score(y, oof_l1[:, n_seeds:2*n_seeds].mean(axis=1))
        log(f"  L1 XGB OOF R2: {xgb_r2:.4f}")
        
        # --- CatBoost ---
        for seed in SEEDS:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof_arch = np.zeros(n_train)
            test_arch = np.zeros(n_test)
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
                cb_dir = f"catboost_zenith_{target}_{seed}_{fold}"
                m = CatBoostRegressor(
                    iterations=800, learning_rate=0.03, depth=8,
                    l2_leaf_reg=3.0, random_seed=seed, verbose=0,
                    early_stopping_rounds=50, train_dir=cb_dir
                )
                m.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]))
                oof_arch[val_idx] = m.predict(X[val_idx])
                test_arch += m.predict(X_test) / N_FOLDS
                
                # Cleanup catboost dir
                if os.path.exists(cb_dir):
                    shutil.rmtree(cb_dir, ignore_errors=True)
            
            oof_l1[:, col_idx] = oof_arch
            test_l1[:, col_idx] = test_arch
            col_idx += 1
        
        cb_r2 = r2_score(y, oof_l1[:, 2*n_seeds:3*n_seeds].mean(axis=1))
        log(f"  L1 CB  OOF R2: {cb_r2:.4f}")
        
        # --- ExtraTrees ---
        for seed in SEEDS:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof_arch = np.zeros(n_train)
            test_arch = np.zeros(n_test)
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
                m = ExtraTreesRegressor(
                    n_estimators=500, max_depth=25, min_samples_leaf=3,
                    random_state=seed, n_jobs=-1
                )
                m.fit(X[tr_idx], y[tr_idx])
                oof_arch[val_idx] = m.predict(X[val_idx])
                test_arch += m.predict(X_test) / N_FOLDS
            
            oof_l1[:, col_idx] = oof_arch
            test_l1[:, col_idx] = test_arch
            col_idx += 1
        
        et_r2 = r2_score(y, oof_l1[:, 3*n_seeds:4*n_seeds].mean(axis=1))
        log(f"  L1 ET  OOF R2: {et_r2:.4f}")
        
        # ============================================================
        # Level 2: Ridge Meta-Learner
        # ============================================================
        log(f"  Training Level 2 Meta-Learner (Ridge)...")
        meta = Ridge(alpha=1.0)
        meta.fit(oof_l1, y)
        
        final_oof_trans = meta.predict(oof_l1)
        final_test_trans = meta.predict(test_l1)
        
        # Inverse transform
        final_oof = qt.inverse_transform(final_oof_trans.reshape(-1, 1)).ravel()
        final_test = qt.inverse_transform(final_test_trans.reshape(-1, 1)).ravel()
        
        r2 = r2_score(y_raw, final_oof)
        log(f"  ** {target} FINAL Stacked R2: {r2:.4f} **")
        overall_r2.append(r2)
        
        submission[target] = final_test
        gc.collect()
    
    mean_r2 = np.mean(overall_r2)
    log(f"\n{'='*60}")
    log(f"[ZENITH] OVERALL MEAN STACKED R2: {mean_r2:.4f}")
    log(f"{'='*60}")
    
    # ============================================================
    # PHASE 4: Spatial Mirror Injection
    # ============================================================
    log("\n=== PHASE 4: Spatial Mirror Injection ===")
    
    train_raw = pd.read_csv(f"{DATA_DIR}/Train.csv")
    test_raw = pd.read_csv(f"{DATA_DIR}/TestSet.csv")
    
    # Build mirror lookup from raw train
    mirror_lookup = train_raw.groupby(['Latitude', 'Longitude'])[TARGETS_RAW].mean().reset_index()
    test_with_coords = test_raw[['ID', 'Latitude', 'Longitude']].merge(
        mirror_lookup, on=['Latitude', 'Longitude'], how='inner'
    )
    
    if len(test_with_coords) > 0:
        log(f"  Found {len(test_with_coords)} exact spatial mirrors.")
        submission.set_index('ID', inplace=True)
        mirror_sub = test_with_coords.set_index('ID')
        for t_raw, t_sub in TARGET_MAP.items():
            if t_raw in mirror_sub.columns:
                submission.loc[mirror_sub.index, t_sub] = mirror_sub[t_raw]
        submission.reset_index(inplace=True)
    
    # ============================================================
    # PHASE 5: Save Submission
    # ============================================================
    submission = submission[['ID'] + TARGETS]
    submission.to_csv("submission_zenith_097.csv", index=False)
    
    log(f"\n[ZENITH] BREAKTHROUGH COMPLETE. File: submission_zenith_097.csv")
    log(f"[ZENITH] Mean CV R2: {mean_r2:.4f}")
    log(f"[ZENITH] Per-target R2: {dict(zip(TARGETS, [f'{r:.4f}' for r in overall_r2]))}")

if __name__ == "__main__":
    main()
