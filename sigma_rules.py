import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ----------------------------------------
# Step 1 - Load the dataset
# ----------------------------------------
print("Loading dataset...")
df = pd.read_csv("Ransomware_Data.csv")

# Convert labels to numbers
df['label'] = df['Ware Type'].map({'good': 0, 'ransom': 1})

print("Dataset loaded successfully!")
print(f"Total entries: {len(df)}")
print("")

# ----------------------------------------
# Step 2 - Implement Sigma Rule Logic
# ----------------------------------------
# Each rule below is based on published SigmaHQ rules for
# ransomware and suspicious Windows process behaviour.
# A detection fires (1) if ANY rule matches, otherwise benign (0)

print("Applying Sigma rules...")

def apply_sigma_rules(row):

    # Rule 1 - Ransomware File Activity
    # Flags entries showing high file deletion and creation
    # activity combined with suspicious file paths
    # Based on: SigmaHQ rule win_ransomware_files
    if (row['File_Delete_archived'] >= 1 and
            row['File_created'] >= 1 and
            row['suspicious_path'] == 1):
        return 1

    # Rule 2 - Suspicious Process Creation
    # Flags entries where a non-system executable spawns
    # processes at unusual directory depths
    # Based on: SigmaHQ rule win_susp_process_creation
    if (row['Process_Create'] >= 1 and
            row['system_executable'] == 0 and
            row['directory_depth'] >= 4 and
            row['suspicious_path'] == 1):
        return 1

    # Rule 3 - Suspicious Registry Modification
    # Flags entries with registry value changes combined
    # with suspicious process behaviour
    # Based on: SigmaHQ rule win_susp_registry_persist
    if (row['Registry_value_set'] >= 1 and
            row['process-related'] >= 1 and
            row['parent_is_system_executable'] == 0 and
            row['suspicious_path'] == 1):
        return 1

    # Rule 4 - Suspicious Network Activity
    # Flags entries showing network activity from
    # suspicious paths with high file entropy
    # Based on: SigmaHQ rule win_susp_network_connection
    if (row['network-related'] >= 1 and
            row['suspicious_path'] == 1 and
            row['file_name_entropy'] >= 4.5):
        return 1

    # Rule 5 - Pipe Creation with File Activity
    # Flags suspicious inter-process communication
    # combined with file operations
    # Based on: SigmaHQ rule win_susp_pipe_created
    if (row['Pipe_Created'] >= 1 and
            row['file-related'] >= 1 and
            row['suspicious_path'] == 1):
        return 1

    # No rule matched - classify as benign
    return 0

# Apply all rules to every row in the dataset
df['sigma_prediction'] = df.apply(apply_sigma_rules, axis=1)

print("Sigma rules applied successfully!")
print("")

# ----------------------------------------
# Step 3 - Evaluate Sigma Rule Performance
# ----------------------------------------
y_true = df['label']
y_pred_sigma = df['sigma_prediction']

accuracy = accuracy_score(y_true, y_pred_sigma)
precision = precision_score(y_true, y_pred_sigma)
recall = recall_score(y_true, y_pred_sigma)
f1 = f1_score(y_true, y_pred_sigma)

print("=" * 40)
print("SIGMA RULES PERFORMANCE RESULTS")
print("=" * 40)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
print("=" * 40)
print("")

# ----------------------------------------
# Step 4 - Confusion Matrix
# ----------------------------------------
print("Confusion Matrix:")
print("(Rows = Actual, Columns = Predicted)")
print("(0 = Benign, 1 = Ransomware)")
print("")
cm = confusion_matrix(y_true, y_pred_sigma)
print(cm)
print("")

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives  (Correctly identified benign):     {tn}")
print(f"False Positives (Benign flagged as ransomware):    {fp}")
print(f"False Negatives (Ransomware missed by model):      {fn}")
print(f"True Positives  (Correctly identified ransomware): {tp}")
print("")

# ----------------------------------------
# Step 5 - Full classification report
# ----------------------------------------
print("Full Classification Report:")
print(classification_report(y_true, y_pred_sigma,
      target_names=['Benign', 'Ransomware']))

# ----------------------------------------
# Step 6 - Direct comparison summary
# ----------------------------------------
print("=" * 40)
print("DIRECT COMPARISON SUMMARY")
print("=" * 40)
print(f"{'Metric':<15} {'Random Forest':>15} {'Sigma Rules':>12}")
print("-" * 40)
print(f"{'Accuracy':<15} {'98.15%':>15} {accuracy*100:>11.2f}%")
print(f"{'Precision':<15} {'93.04%':>15} {precision*100:>11.2f}%")
print(f"{'Recall':<15} {'99.37%':>15} {recall*100:>11.2f}%")
print(f"{'F1 Score':<15} {'96.10%':>15} {f1*100:>11.2f}%")
print("=" * 40)