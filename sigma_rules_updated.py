import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# Step 1 - Load the dataset
# ----------------------------------------
print("Loading dataset...")
df = pd.read_csv("Ransomware_Data.csv")
df['label'] = df['Ware Type'].map({'good': 0, 'ransom': 1})
print(f"Dataset loaded: {len(df)} entries")
print("")

# ----------------------------------------
# Step 2 - Original Sigma Rules
# (strict AND conditions)
# ----------------------------------------
print("Applying original Sigma rules...")

def original_sigma_rules(row):
    # Rule 1 - Ransomware File Activity
    if (row['File_Delete_archived'] >= 1 and
            row['File_created'] >= 1 and
            row['suspicious_path'] == 1):
        return 1
    # Rule 2 - Suspicious Process Creation
    if (row['Process_Create'] >= 1 and
            row['system_executable'] == 0 and
            row['directory_depth'] >= 4 and
            row['suspicious_path'] == 1):
        return 1
    # Rule 3 - Suspicious Registry Modification
    if (row['Registry_value_set'] >= 1 and
            row['process-related'] >= 1 and
            row['parent_is_system_executable'] == 0 and
            row['suspicious_path'] == 1):
        return 1
    # Rule 4 - Suspicious Network Activity
    if (row['network-related'] >= 1 and
            row['suspicious_path'] == 1 and
            row['file_name_entropy'] >= 4.5):
        return 1
    # Rule 5 - Pipe Creation with File Activity
    if (row['Pipe_Created'] >= 1 and
            row['file-related'] >= 1 and
            row['suspicious_path'] == 1):
        return 1
    return 0

df['original_pred'] = df.apply(original_sigma_rules, axis=1)
print("Original rules applied!")
print("")

# ----------------------------------------
# Step 3 - Updated Sigma Rules
# (loosened conditions — more realistic)
# ----------------------------------------
print("Applying updated Sigma rules...")

def updated_sigma_rules(row):
    # Rule 1 - Ransomware File Activity
    # Loosened: suspicious path OR file deletion alone
    # is enough — does not require all three
    if (row['File_Delete_archived'] >= 1 and
            row['File_created'] >= 1):
        return 1
    if (row['suspicious_path'] == 1 and
            row['file-related'] >= 1):
        return 1

    # Rule 2 - Suspicious Process Creation
    # Loosened: removed suspicious_path requirement
    if (row['Process_Create'] >= 1 and
            row['system_executable'] == 0 and
            row['directory_depth'] >= 4):
        return 1

    # Rule 3 - Suspicious Registry Modification
    # Loosened: removed parent_is_system_executable
    if (row['Registry_value_set'] >= 1 and
            row['process-related'] >= 1 and
            row['suspicious_path'] == 1):
        return 1

    # Rule 4 - Suspicious Network Activity
    # Loosened: lowered entropy threshold
    if (row['network-related'] >= 1 and
            row['file_name_entropy'] >= 3.0):
        return 1

    # Rule 5 - Pipe Creation with File Activity
    # Loosened: removed suspicious_path requirement
    if (row['Pipe_Created'] >= 1 and
            row['file-related'] >= 1):
        return 1

    # Rule 6 - High File Entropy (NEW)
    # High randomness in file names is a strong
    # ransomware indicator even without other signals
    if row['file_name_entropy'] >= 4.0:
        return 1

    # Rule 7 - Deep Directory with Process Activity (NEW)
    # Ransomware often operates from deep directories
    if (row['directory_depth'] >= 6 and
            row['process-related'] >= 1):
        return 1

    # Rule 8 - Unusual Process to Parent Ratio (NEW)
    # Abnormal spawning rate is a key LotL indicator
    if row['process_vs_parent_freq_ratio'] >= 0.5:
        return 1

    return 0

df['updated_pred'] = df.apply(updated_sigma_rules, axis=1)
print("Updated rules applied!")
print("")

# ----------------------------------------
# Step 4 - Evaluate both rule sets
# ----------------------------------------
y_true = df['label']

# Original results
orig_acc  = accuracy_score(y_true, df['original_pred'])
orig_prec = precision_score(y_true, df['original_pred'])
orig_rec  = recall_score(y_true, df['original_pred'])
orig_f1   = f1_score(y_true, df['original_pred'])

# Updated results
upd_acc  = accuracy_score(y_true, df['updated_pred'])
upd_prec = precision_score(y_true, df['updated_pred'])
upd_rec  = recall_score(y_true, df['updated_pred'])
upd_f1   = f1_score(y_true, df['updated_pred'])

print("=" * 55)
print("ORIGINAL SIGMA RULES RESULTS")
print("=" * 55)
print(f"Accuracy:  {orig_acc*100:.2f}%")
print(f"Precision: {orig_prec*100:.2f}%")
print(f"Recall:    {orig_rec*100:.2f}%")
print(f"F1 Score:  {orig_f1*100:.2f}%")
print("")

cm_orig = confusion_matrix(y_true, df['original_pred'])
tn, fp, fn, tp = cm_orig.ravel()
print(f"True Positives  (ransomware caught):  {tp}")
print(f"False Negatives (ransomware missed):  {fn}")
print(f"False Positives (benign flagged):     {fp}")
print(f"True Negatives  (benign correct):     {tn}")
print("")

print("=" * 55)
print("UPDATED SIGMA RULES RESULTS")
print("=" * 55)
print(f"Accuracy:  {upd_acc*100:.2f}%")
print(f"Precision: {upd_prec*100:.2f}%")
print(f"Recall:    {upd_rec*100:.2f}%")
print(f"F1 Score:  {upd_f1*100:.2f}%")
print("")

cm_upd = confusion_matrix(y_true, df['updated_pred'])
tn2, fp2, fn2, tp2 = cm_upd.ravel()
print(f"True Positives  (ransomware caught):  {tp2}")
print(f"False Negatives (ransomware missed):  {fn2}")
print(f"False Positives (benign flagged):     {fp2}")
print(f"True Negatives  (benign correct):     {tn2}")
print("")

print("=" * 65)
print("COMPARISON — ORIGINAL vs UPDATED SIGMA RULES")
print("=" * 65)
print(f"{'Metric':<12} {'Original':>12} {'Updated':>12} {'Change':>12}")
print("-" * 65)
metrics_list = [
    ('Accuracy',  orig_acc,  upd_acc),
    ('Precision', orig_prec, upd_prec),
    ('Recall',    orig_rec,  upd_rec),
    ('F1 Score',  orig_f1,   upd_f1),
]
for name, orig, upd in metrics_list:
    change = (upd - orig) * 100
    direction = "↑" if change > 0 else "↓"
    print(f"{name:<12} {orig*100:>11.2f}% {upd*100:>11.2f}% "
          f"{direction}{abs(change):>10.2f}%")
print("=" * 65)
print("")

# ----------------------------------------
# Step 5 - Save comparison chart
# ----------------------------------------
print("Saving Sigma rules comparison chart...")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
orig_scores = [orig_acc*100, orig_prec*100,
               orig_rec*100, orig_f1*100]
upd_scores  = [upd_acc*100, upd_prec*100,
               upd_rec*100, upd_f1*100]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width/2, orig_scores, width,
               label='Original Sigma Rules (Strict)',
               color='#e74c3c', alpha=0.85)
bars2 = ax.bar(x + width/2, upd_scores, width,
               label='Updated Sigma Rules (Loosened)',
               color='#f39c12', alpha=0.85)

ax.set_ylim(0, 110)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_xlabel('Metric', fontsize=11)
ax.set_title('Sigma Rules Comparison\nOriginal (Strict) vs Updated (Loosened) Conditions',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.2f}%',
            ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#c0392b')

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.2f}%',
            ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#d35400')

plt.tight_layout()
plt.savefig("sigma_rules_comparison.png", dpi=150,
            bbox_inches='tight')
print("Chart saved as sigma_rules_comparison.png")
print("")
print("Sigma rules analysis complete!")