import pandas as pd
import numpy as np

# This script generates a 'Clean' submission from Stage 3 results
# Removing Mirroring and Reconciliation to find the baseline performance.

def fix_submission():
    raw = pd.read_csv("ascension_stage3_preds_raw.csv")
    targets = ['Target_Al', 'Target_Ca', 'Target_Mg', 'Target_K', 'Target_N', 
               'Target_Cu', 'Target_Fe', 'Target_Mn', 'Target_P', 'Target_B', 
               'Target_S', 'Target_Zn', 'Target_Na']
    
    # Just the raw blender results
    final = raw[['ID'] + targets]
    final.to_csv("submission_ascension_clean.csv", index=False)
    print("Generated submission_ascension_clean.csv (No Mirrors, No Reconciliation)")

if __name__ == "__main__":
    fix_submission()
