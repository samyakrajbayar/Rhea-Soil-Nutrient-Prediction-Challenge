import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
import os

def run_pseudo_labeling():
    print("Loading all feature sets for Pseudo-Labeling Stacker...")
    train_proc = pd.read_csv("train_processed.csv")
    test_proc = pd.read_csv("test_processed.csv")
    train_dae = pd.read_csv("train_dae_features.csv")
    test_dae = pd.read_csv("test_dae_features.csv")
    train_mlp = pd.read_csv("train_mlp_preds.csv")
    test_mlp = pd.read_csv("test_mlp_preds.csv")
    
    # Merge all
    train = train_proc.merge(train_dae, on='ID').merge(train_mlp, on='ID')
    test = test_proc.merge(test_dae, on='ID').merge(test_mlp, on='ID')
    
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    # Common features (Intersection of all base features)
    common_cols = list(set(train_proc.columns).intersection(set(test_proc.columns)))
    features_base = [c for c in common_cols if c not in ['ID'] + targets]
    
    # Add DAE latent features
    dae_features = [c for c in train_dae.columns if c != 'ID']
    # Add MLP signatures
    mlp_features = [c for c in train_mlp.columns if c != 'ID']
    
    features = features_base + dae_features + mlp_features
    
    print(f"Using {len(features)} total features for Pseudo-Labeling Stacker.")
    
    test_preds = pd.DataFrame(index=test.index)
    test_preds['ID'] = test['ID']
    
    overall_r2 = []
    
    for target in targets:
        print(f"\n--- Optimizing {target} with Pseudo-Labeling ---")
        
        valid_idx = train[train[target].notnull()].index
        X_train = train.loc[valid_idx, features]
        y_train = train.loc[valid_idx, target].values.reshape(-1, 1)
        X_test = test[features]
        
        # Transform target
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        y_train_trans = qt.fit_transform(y_train).ravel()
        
        # 1. Initial Model (3-fold CV)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof_initial = np.zeros(len(X_train))
        test_initial = np.zeros(len(X_test))
        
        lgb_params = {
            'n_estimators': 1500,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train_trans[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train_trans[val_idx]
            
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
            
            oof_initial[val_idx] = model.predict(X_val)
            test_initial += model.predict(X_test) / 3.0
            
        initial_r2 = r2_score(y_train_trans, oof_initial)
        print(f"Initial CV R2 for {target}: {initial_r2:.4f}")
        
        # 2. Pseudo-Labeling Round
        # We take ALL test predictions as labels to guide the manifold
        X_augmented = pd.concat([X_train, X_test], axis=0)
        y_augmented = np.concatenate([y_train_trans, test_initial])
        
        # Retrain on augmented data
        final_test_pred = np.zeros(len(X_test))
        for seed in [42, 2024, 777]:
            model = lgb.LGBMRegressor(**lgb_params)
            model.set_params(random_state=seed)
            model.fit(X_augmented, y_augmented)
            final_test_pred += model.predict(X_test) / 3.0
            
        final_test_orig = qt.inverse_transform(final_test_pred.reshape(-1, 1)).ravel()
        test_preds[target] = final_test_orig
        
        overall_r2.append(initial_r2)
        
    print(f"\nFinal Estimated Mean R2 (Proxy): {np.mean(overall_r2):.4f}")
    test_preds.to_csv("final_submission_097.csv", index=False)
    print("Final Pseudo-Labeled submission saved.")

if __name__ == "__main__":
    run_pseudo_labeling()
