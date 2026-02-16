import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
import os
import sys

# Define path
DATA_PATH = r"c:\Users\Administrator.RAJESH\Desktop\22\New folder (6)\rhea-soil-nutrient-prediction-challenge20260210-28069-1fmhbqj"

def train():
    # Setup logging
    log_file = open("training.log", "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("Loading preprocessed data...")
    train = pd.read_csv("train_processed.csv")
    test = pd.read_csv("test_processed.csv")
    
    targets = ['Target_Al', 'Target_B', 'Target_Ca', 'Target_Cu', 'Target_Fe', 
               'Target_Mg', 'Target_Mn', 'Target_N', 'Target_P', 'Target_K', 
               'Target_Na', 'Target_S', 'Target_Zn']
    
    test_preds = pd.DataFrame(index=test.index)
    test_preds['ID'] = test['ID']
    
    features_base = ['Latitude', 'Longitude', 'Depth_cm', 'cluster',
                     'lat_lon', 'lat_sq', 'lon_sq', 'lat_depth', 'lon_depth',
                     'rot_45_x', 'rot_45_y', 'month_sin', 'month_cos', 
                     'day_sin', 'day_cos', 'year']
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42) # Faster CV for stacking
    overall_rmse = []
    overall_r2 = []
    
    total_targets = len(targets)
    for i, target in enumerate(targets):
        progress = (i / total_targets) * 100
        log(f"\n--- [{progress:.1f}%] Training KNN-Enhanced Stack for {target} ---")
        
        # Select features: Base + Nutrient-specific KNN
        current_features = features_base + [f'knn_mean_{target}']
        
        valid_idx = train[train[target].notnull()].index
        X = train.loc[valid_idx, current_features]
        y = train.loc[valid_idx, target].values.reshape(-1, 1)
        
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        y_trans = qt.fit_transform(y).ravel()
        
        oof_lgbm = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))
        oof_cb = np.zeros(len(X))
        
        test_lgbm = np.zeros(len(test))
        test_xgb = np.zeros(len(test))
        test_cb = np.zeros(len(test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_trans)):
            X_tr, y_tr = X.iloc[train_idx], y_trans[train_idx]
            X_val, y_val = X.iloc[val_idx], y_trans[val_idx]
            
            # Fast Tier 1 Models
            model_lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.1, num_leaves=31, 
                                           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, 
                                           random_state=42, n_jobs=-1, verbose=-1)
            model_lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30)])
            oof_lgbm[val_idx] = model_lgbm.predict(X_val)
            test_lgbm += model_lgbm.predict(test[current_features]) / 3.0
            
            model_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=4, 
                                          subsample=0.8, colsample_bytree=0.8, 
                                          random_state=42, n_jobs=-1, tree_method='hist',
                                          early_stopping_rounds=30)
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_xgb[val_idx] = model_xgb.predict(X_val)
            test_xgb += model_xgb.predict(test[current_features]) / 3.0
            
            model_cb = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=4, l2_leaf_reg=3, 
                                         random_seed=42, verbose=0, early_stopping_rounds=30)
            model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            oof_cb[val_idx] = model_cb.predict(X_val)
            test_cb += model_cb.predict(test[current_features]) / 3.0
            
            log(f"Fold {fold+1} complete.")

        # Tier 2 Stacking
        oof_combined = np.column_stack([oof_lgbm, oof_xgb, oof_cb])
        test_combined = np.column_stack([test_lgbm, test_xgb, test_cb])
        
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(oof_combined, y_trans)
        
        final_oof_trans = meta_model.predict(oof_combined)
        final_test_trans = meta_model.predict(test_combined)
        
        final_oof = qt.inverse_transform(final_oof_trans.reshape(-1, 1)).ravel()
        final_test = qt.inverse_transform(final_test_trans.reshape(-1, 1)).ravel()
        
        y_orig = y.ravel()
        rmse = np.sqrt(mean_squared_error(y_orig, final_oof))
        r2 = r2_score(y_orig, final_oof)
        log(f"Target {target} Stats -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
        overall_rmse.append(rmse)
        overall_r2.append(r2)
        
        test_preds[target] = final_test
        
    log(f"\n[100.0%] Final Overall -> RMSE: {np.mean(overall_rmse):.4f}, R2: {np.mean(overall_r2):.4f}")
    test_preds.to_csv("test_predictions_optimized.csv", index=False)
    log_file.close()

if __name__ == "__main__":
    train()
