import pandas as pd
import numpy as np

# Project Ascension: Pure Recovery
# This script removes the 'Global Reconciliation' trap which poisoned the 0.769 submission.
# It preserves the Stage 3 Multi-Architecture Blend and applies only the high-confidence Spatial Mirroring.

def generate_pure_submission():
    print("Generating Pure Ascension Submission...")
    
    # 1. Load Raw Stage 3 Predictions (Blended LGB+XGB+CB)
    raw = pd.read_csv("ascension_stage3_preds_raw.csv")
    
    # 2. Load Spatial Mirrors
    mirrors = pd.read_csv("ascension_mirrors.csv")
    
    targets = ['Target_Al', 'Target_Ca', 'Target_Mg', 'Target_K', 'Target_N', 
               'Target_Cu', 'Target_Fe', 'Target_Mn', 'Target_P', 'Target_B', 
               'Target_S', 'Target_Zn', 'Target_Na']
    
    final_sub = raw.copy()
    
    # 3. Inject Mirrors (Explicit matching IDs)
    print(f"Injecting {len(mirrors)} Spatial Mirrors...")
    final_sub.set_index('ID', inplace=True)
    mirrors.set_index('ID', inplace=True)
    final_sub.update(mirrors)
    final_sub.reset_index(inplace=True)
    
    # 4. Save Final Breakthrough Submission (Pure)
    final_sub = final_sub[['ID'] + targets]
    final_sub.to_csv("submission_ascension_pure.csv", index=False)
    print("Success: submission_ascension_pure.csv generated.")

if __name__ == "__main__":
    generate_pure_submission()
