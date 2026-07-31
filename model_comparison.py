import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------
# Step 1 - Load and prepare data
# ----------------------------------------
print("Loading dataset...")
df = pd.read_csv("Ransomware_Data.csv")
df['label'] = df['Ware Type'].map({'good': 0, 'ransom': 1})
X = df.drop(columns=['Ware Type', 'label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} rows")
print(f"Testing set:  {X_test.shape[0]} rows")
print("")

# ----------------------------------------
# Step 2 - Define models
# ----------------------------------------
models = {
    'Decision Tree': DecisionTreeClassifier(
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

# ----------------------------------------
# Step 3 - Train and evaluate each model
# ----------------------------------------
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name] = {
        'Accuracy':  acc,
        'Precision': prec,
        'Recall':    rec,
        'F1 Score':  f1,
        'y_pred':    y_pred
    }

    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print("")

# ----------------------------------------
# Step 4 - Print comparison table
# ----------------------------------------
print("=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(f"{'Metric':<12}", end="")
for name in models.keys():
    print(f"{name:>20}", end="")
print()
print("-" * 60)

for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    print(f"{metric:<12}", end="")
    for name in models.keys():
        val = results[name][metric] * 100
        print(f"{val:>19.2f}%", end="")
    print()

print("=" * 60)
print("")

# ----------------------------------------
# Step 5 - Print classification reports
# ----------------------------------------
for name in models.keys():
    print(f"Classification Report — {name}:")
    print(classification_report(
        y_test, results[name]['y_pred'],
        target_names=['Benign', 'Ransomware']
    ))
    print("")

# ----------------------------------------
# Step 6 - Save comparison bar chart
# ----------------------------------------
print("Saving comparison chart...")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
model_names = list(models.keys())
colours = ['#e74c3c', '#f39c12', '#2ecc71']

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))

for i, (name, colour) in enumerate(zip(model_names, colours)):
    values = [results[name][m] * 100 for m in metrics]
    bars = ax.bar(x + i * width, values, width,
                  label=name, color=colour, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val:.1f}%',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold')

ax.set_ylim(0, 115)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_xlabel('Metric', fontsize=11)
ax.set_title('Machine Learning Model Comparison\nDecision Tree vs Logistic Regression vs Random Forest',
             fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=90, color='grey', linestyle='--',
           alpha=0.5, linewidth=1)
ax.text(3.4, 90.5, '90% threshold',
        fontsize=8, color='grey')

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150,
            bbox_inches='tight')
print("Chart saved as model_comparison.png")
print("")
print("Model comparison complete!")