import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import warnings, os, gc
warnings.filterwarnings('ignore')

# ============================================================
# PROJECT ZENITH: LITE SPEED-RUN (0.90+ Goal in ~1 Hour)
# ============================================================
# 1 Seed, 3 Folds, 3 Architectures
# ============================================================

DATA_DIR = "rhea-soil-nutrient-prediction-challenge20260210-28069-1fmhbqj"
TARGETS = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe',
           'Target_K', 'Target_Mg', 'Target_Mn', 'Target_N', 'Target_Na',
           'Target_P', 'Target_S', 'Target_Zn']

N_FOLDS = 3
SEEDS = [42]
LOG_FILE = "zenith_lite.log"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    log("=" * 60)
    log("PROJECT ZENITH: LITE SPEED-RUN")
    log("=" * 60)
    
    log("\n=== PHASE 1: Loading Data ===")
    train = pd.read_csv("train_infinite.csv")
    test = pd.read_csv("test_infinite.csv")
    log(f"  Train: {train.shape}, Test: {test.shape}")
    
    # Load DAE/MLP if available
    for f_train, f_test, name in [("train_dae_features.csv", "test_dae_features.csv", "DAE"),
                                  ("train_mlp_preds.csv", "test_mlp_preds.csv", "MLP")]:
        if os.path.exists(f_train):
            log(f"  Loading {name} signatures...")
            train = train.merge(pd.read_csv(f_train), on='ID', how='left')
            test = test.merge(pd.read_csv(f_test), on='ID', how='left')
    
    common_cols = sorted(list(set(train.columns) & set(test.columns)))
    feature_cols = [c for c in common_cols if c not in ['ID'] + TARGETS]
    log(f"\n=== PHASE 2: Feature Selection ({len(feature_cols)} features) ===")
    
    for c in feature_cols:
        med = train[c].median()
        train[c] = train[c].fillna(med if not pd.isna(med) else 0)
        test[c] = test[c].fillna(med if not pd.isna(med) else 0)
    
    log("\n=== PHASE 3: High-Speed Stacking Ensemble ===")
    submission = pd.DataFrame({'ID': test['ID']})
    
    for target in TARGETS:
        log(f"\n--- TARGET: {target} ---")
        valid_mask = train[target].notnull()
        X = train.loc[valid_mask, feature_cols].values
        y_raw = train.loc[valid_mask, target].values
        X_test = test[feature_cols].values
        
        qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(1000, len(y_raw)))
        y = qt.fit_transform(y_raw.reshape(-1, 1)).ravel()
        
        n_archs = 3
        oof_l1 = np.zeros((len(y), n_archs))
        test_l1 = np.zeros((len(X_test), n_archs))
        
        # 1. LightGBM
        log("  Training LGBM...")
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
            model.fit(X[train_idx], y[train_idx])
            oof_l1[val_idx, 0] = model.predict(X[val_idx])
            test_l1[:, 0] += model.predict(X_test) / N_FOLDS
        log(f"    L1 LGB R2: {r2_score(y, oof_l1[:, 0]):.4f}")
        
        # 2. XGBoost
        log("  Training XGBoost...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
            model.fit(X[train_idx], y[train_idx])
            oof_l1[val_idx, 1] = model.predict(X[val_idx])
            test_l1[:, 1] += model.predict(X_test) / N_FOLDS
        log(f"    L1 XGB R2: {r2_score(y, oof_l1[:, 1]):.4f}")
        
        # 3. CatBoost
        log("  Training CatBoost...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_state=42, verbose=0, thread_count=-1)
            model.fit(X[train_idx], y[train_idx])
            oof_l1[val_idx, 2] = model.predict(X[val_idx])
            test_l1[:, 2] += model.predict(X_test) / N_FOLDS
        log(f"    L1 CB R2: {r2_score(y, oof_l1[:, 2]):.4f}")
        
        # Meta-Learner
        meta = Ridge(alpha=1.0)
        meta.fit(oof_l1, y)
        final_preds_norm = meta.predict(test_l1)
        final_preds = qt.inverse_transform(final_preds_norm.reshape(-1, 1)).ravel()
        submission[target] = final_preds
        log(f"  ** {target} Stacked R2: {r2_score(y, meta.predict(oof_l1)):.4f} **")
        
        gc.collect()

    sub_file = "submission_zenith_lite.csv"
    submission.to_csv(sub_file, index=False)
    log(f"\n[SUCCESS] Final submission saved to {sub_file}")

if __name__ == "__main__":
    main()
