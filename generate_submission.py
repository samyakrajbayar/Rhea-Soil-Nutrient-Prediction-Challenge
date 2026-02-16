import pandas as pd
import os

DATA_PATH = r"c:\Users\Administrator.RAJESH\Desktop\22\New folder (6)\rhea-soil-nutrient-prediction-challenge20260210-28069-1fmhbqj"

def generate():
    print("Generating submission...")
    # Load sample submission and predictions
    sample = pd.read_csv(os.path.join(DATA_PATH, "SampleSubmission.csv"))
    preds = pd.read_csv("test_predictions_optimized.csv")
    mask = pd.read_csv(os.path.join(DATA_PATH, "TargetPred_To_Keep.csv"))
    
    # Columns to process (Targets)
    targets = [col for col in sample.columns if col != 'ID']
    
    # Create final submission dataframe
    sub = sample.copy()
    
    # Align mask and preds by ID index for easier multiplication
    preds.set_index('ID', inplace=True)
    mask.set_index('ID', inplace=True)
    sub.set_index('ID', inplace=True)
    
    for target_col in targets:
        # target_col is like 'Target_Al'
        # mask_col is like 'Al'
        mask_col = target_col.replace('Target_', '')
        
        # Multiply predictions by mask
        # sub[target_col] will be updated
        if mask_col in mask.columns and target_col in preds.columns:
            # Note: sub[target_col] = preds[target_col] * mask[mask_col]
            # We align by index (ID)
            sub[target_col] = preds[target_col] * mask[mask_col]
        else:
            print(f"Warning: {target_col} or {mask_col} missing in files.")

    # Reset index to get ID column back
    sub.reset_index(inplace=True)
    
    # Save submission
    sub.to_csv("final_submission.csv", index=False)
    print("Final submission saved to final_submission.csv")
    print("Submission shape:", sub.shape)
    print("Sample check:")
    print(sub.head())

if __name__ == "__main__":
    generate()
