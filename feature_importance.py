import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# Step 1 - Load the saved model
# ----------------------------------------
print("Loading trained Random Forest model...")

with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")
print("")

# ----------------------------------------
# Step 2 - Extract feature importances
# ----------------------------------------
# Get the feature names from the CSU dataset
feature_names = [
    'File_Delete_archived', 'File_created', 'File_creation_time_changed',
    'Pipe_Created', 'Process_Create', 'Registry_value_set',
    'process-related', 'network-related', 'file-related',
    'suspicious_path', 'system_executable', 'path_length',
    'directory_depth', 'process_name_length',
    'process_vs_parent_freq_ratio', 'executable_depth_diff',
    'parent_is_system_executable', 'extension_similarity',
    'file_name_entropy'
]

# Extract importance scores from the model
importances = model.feature_importances_

# Create a dataframe for easy viewing and sorting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# ----------------------------------------
# Step 3 - Print results
# ----------------------------------------
print("=" * 50)
print("FEATURE IMPORTANCE RANKINGS")
print("(Higher score = more important for detection)")
print("=" * 50)
for i, row in importance_df.iterrows():
    bar = "█" * int(row['Importance'] * 200)
    print(f"{row['Feature']:<35} {row['Importance']:.4f}  {bar}")
print("=" * 50)
print("")

# ----------------------------------------
# Step 4 - Save as a chart
# ----------------------------------------
print("Saving feature importance chart...")

fig, ax = plt.subplots(figsize=(10, 8))

colours = ['#2ecc71' if imp > 0.1 else '#3498db' if imp > 0.05
           else '#95a5a6' for imp in importance_df['Importance']]

bars = ax.barh(
    importance_df['Feature'],
    importance_df['Importance'],
    color=colours
)

ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Random Forest Feature Importance\nCSU Ransomware Dataset — Sysmon Log Features',
             fontsize=13, fontweight='bold')
ax.invert_yaxis()

# Add value labels on bars
for bar, val in zip(bars, importance_df['Importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='High importance (>0.10)'),
    Patch(facecolor='#3498db', label='Medium importance (0.05-0.10)'),
    Patch(facecolor='#95a5a6', label='Lower importance (<0.05)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
print("Chart saved as feature_importance.png")
print("")
print("Feature importance analysis complete!")