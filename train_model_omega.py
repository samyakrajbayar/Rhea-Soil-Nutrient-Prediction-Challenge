import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
import os

def train_omega():
    log_file = open("omega_training.log", "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("Loading Omega data...")
    train = pd.read_csv("train_omega.csv")
    test = pd.read_csv("test_omega.csv")
    
    # Define groups by expected predictability (based on previous runs)
    group_a = ['Target_Al', 'Target_Ca', 'Target_Mg', 'Target_K', 'Target_N', 'Target_Cu']
    group_b = ['Target_Fe', 'Target_Mn']
    group_c = ['Target_P', 'Target_B', 'Target_S', 'Target_Zn', 'Target_Na']
    
    all_targets = group_a + group_b + group_c
    
    # Base features (Intersection of common columns)
    common_cols = list(set(train.columns).intersection(set(test.columns)))
    features_base = [c for c in common_cols if c not in ['ID'] + all_targets and not c.startswith('tnn_')]
    
    # Target-specific TNN features are safe because they are neighbors' values
    tnn_features = [c for c in train.columns if c.startswith('tnn_')]
    
    test_preds = pd.DataFrame(index=test.index)
    test_preds['ID'] = test['ID']
    
    # Storing OOF and Test predictions to use as features for subsequent groups
    chain_features_train = pd.DataFrame(index=train.index)
    chain_features_test = pd.DataFrame(index=test.index)
    
    overall_r2 = []
    
    def train_group(target_list, current_features):
        for target in target_list:
            log(f"\n--- Training Hierarchical Model for {target} ---")
            
            valid_idx = train[train[target].notnull()].index
            X = train.loc[valid_idx, current_features]
            y = train.loc[valid_idx, target].values.reshape(-1, 1)
            
            # Normalization
            qt = QuantileTransformer(output_distribution='normal', random_state=42)
            y_trans = qt.fit_transform(y).ravel()
            
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            oof_final = np.zeros(len(X))
            test_final = np.zeros(len(test))
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_trans)):
                X_tr, y_tr = X.iloc[tr_idx], y_trans[tr_idx]
                X_val, y_val = X.iloc[val_idx], y_trans[val_idx]
                
                # Multi-model blend for high fidelity
                # LGBM
                model_lgb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, 
                                             feature_fraction=0.8, bagging_fraction=0.8, random_state=42, verbose=-1)
                model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
                
                # XGB
                model_xgb = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, 
                                             subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method='hist')
                model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
                # Blend
                p1 = model_lgb.predict(X_val)
                p2 = model_xgb.predict(X_val)
                fold_pred = (p1 + p2) / 2.0
                oof_final[val_idx] = fold_pred
                
                t1 = model_lgb.predict(test[current_features])
                t2 = model_xgb.predict(test[current_features])
                test_final += (t1 + t2) / (2.0 * 3.0)
                
                fold_r2 = r2_score(y_val, fold_pred)
                log(f"  Fold {fold+1} Blend R2 (Transformed): {fold_r2:.4f}")
                
            # Inverse transform
            final_oof_orig = qt.inverse_transform(oof_final.reshape(-1, 1)).ravel()
            final_test_orig = qt.inverse_transform(test_final.reshape(-1, 1)).ravel()
            
            r2 = r2_score(y.ravel(), final_oof_orig)
            log(f"Target {target} R2: {r2:.4f}")
            overall_r2.append(r2)
            
            # Save for chain (using OOF for train to prevent leakage)
            chain_features_train.loc[valid_idx, f'pred_{target}'] = oof_final
            chain_features_test[f'pred_{target}'] = test_final
            
            # Store final result
            test_preds[target] = final_test_orig

    # Phase 1: Group A (High Confidence)
    # Give Group A access to TNN and Base features
    train_group(group_a, features_base + tnn_features)
    
    # Phase 2: Group B (Intermediate)
    # Give Group B access to Base + TNN + Group A Preds
    features_b = features_base + tnn_features + [f'pred_{t}' for t in group_a]
    # Update train with OOF preds
    current_train = pd.concat([train, chain_features_train], axis=1)
    current_test = pd.concat([test, chain_features_test], axis=1)
    
    # Helper to re-wrap train_group with new data
    def train_group_omega(target_list, features_list, df_train, df_test):
        for target in target_list:
            log(f"\n--- Training Hierarchical Model (Omega) for {target} ---")
            valid_idx = df_train[df_train[target].notnull()].index
            X = df_train.loc[valid_idx, features_list]
            y = df_train.loc[valid_idx, target].values.reshape(-1, 1)
            qt = QuantileTransformer(output_distribution='normal', random_state=42)
            y_trans = qt.fit_transform(y).ravel()
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            oof_final = np.zeros(len(X))
            test_final = np.zeros(len(df_test))
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_trans)):
                X_tr, y_tr = X.iloc[tr_idx], y_trans[tr_idx]
                X_val, y_val = X.iloc[val_idx], y_trans[val_idx]
                model_lgb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, feature_fraction=0.8, random_state=42, verbose=-1)
                model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])
                p_val = model_lgb.predict(X_val)
                oof_final[val_idx] = p_val
                test_final += model_lgb.predict(df_test[features_list]) / 3.0
            final_oof_orig = qt.inverse_transform(oof_final.reshape(-1, 1)).ravel()
            final_test_orig = qt.inverse_transform(test_final.reshape(-1, 1)).ravel()
            r2 = r2_score(y.ravel(), final_oof_orig)
            log(f"Target {target} R2: {r2:.4f}")
            overall_r2.append(r2)
            chain_features_train.loc[valid_idx, f'pred_{target}'] = oof_final
            chain_features_test[f'pred_{target}'] = test_final
            test_preds[target] = final_test_orig

    train_group_omega(group_b, features_b, current_train, current_test)
    
    # Phase 3: Group C (Low Confidence - Trace Elements)
    # Give Group C access to ALL previous preds
    features_c = features_base + tnn_features + [f'pred_{t}' for t in group_a + group_b]
    current_train = pd.concat([train, chain_features_train], axis=1)
    current_test = pd.concat([test, chain_features_test], axis=1)
    train_group_omega(group_c, features_c, current_train, current_test)
    
    log(f"\n[Project Omega] Final Mean R2: {np.mean(overall_r2):.4f}")
    test_preds.to_csv("submission_omega.csv", index=False)
    log_file.close()

if __name__ == "__main__":
    train_omega()
