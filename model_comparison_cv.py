import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time

# ----------------------------------------
# Step 1 - Load and prepare data
# ----------------------------------------
print("Loading dataset...")
df = pd.read_csv("Ransomware_Data.csv")
df['label'] = df['Ware Type'].map({'good': 0, 'ransom': 1})
X = df.drop(columns=['Ware Type', 'label'])
y = df['label']

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print("")

# ----------------------------------------
# Step 2 - Define models and scoring
# ----------------------------------------
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)
}

scoring = {
    'accuracy':  make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall':    make_scorer(recall_score),
    'f1':        make_scorer(f1_score)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------------------
# Step 3 - Run cross validation for each
# ----------------------------------------
cv_results = {}

for name, model in models.items():
    print(f"Running 5-fold cross validation for {name}...")
    print("Please wait...")
    start = time.time()

    results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )

    elapsed = (time.time() - start) / 60
    print(f"Completed in {elapsed:.1f} minutes!")
    print("")

    cv_results[name] = {
        'Accuracy':  results['test_accuracy'],
        'Precision': results['test_precision'],
        'Recall':    results['test_recall'],
        'F1 Score':  results['test_f1']
    }

# ----------------------------------------
# Step 4 - Print results per model
# ----------------------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for name in models.keys():
    print("=" * 55)
    print(f"CROSS VALIDATION RESULTS — {name}")
    print("=" * 55)
    print(f"{'Fold':<6}", end="")
    for m in metrics:
        print(f"{m:>12}", end="")
    print()
    print("-" * 55)

    for fold in range(5):
        print(f"  {fold+1:<4}", end="")
        for m in metrics:
            val = cv_results[name][m][fold] * 100
            print(f"{val:>11.2f}%", end="")
        print()

    print("-" * 55)
    print(f"{'Mean':<6}", end="")
    for m in metrics:
        val = cv_results[name][m].mean() * 100
        print(f"{val:>11.2f}%", end="")
    print()

    print(f"{'Std':<6}", end="")
    for m in metrics:
        val = cv_results[name][m].std() * 100
        print(f"{val:>11.2f}%", end="")
    print()
    print("")

# ----------------------------------------
# Step 5 - Summary comparison table
# ----------------------------------------
print("=" * 65)
print("CROSS VALIDATION SUMMARY — ALL MODELS")
print("=" * 65)
print(f"{'Metric':<12} {'Model':<22} {'Mean':>8} {'Std':>8}")
print("-" * 65)

for m in metrics:
    for name in models.keys():
        mean = cv_results[name][m].mean() * 100
        std  = cv_results[name][m].std() * 100
        print(f"{m:<12} {name:<22} {mean:>7.2f}% {std:>7.2f}%")
    print()

print("=" * 65)

# ----------------------------------------
# Step 6 - Save cross validation chart
# ----------------------------------------
print("Saving cross validation comparison chart...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colours = {
    'Decision Tree':      '#e74c3c',
    'Logistic Regression': '#f39c12',
    'Random Forest':       '#2ecc71'
}

fold_numbers = [1, 2, 3, 4, 5]

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    for name in models.keys():
        values = cv_results[name][metric] * 100
        mean   = values.mean()
        std    = values.std()
        colour = colours[name]

        ax.plot(fold_numbers, values,
                marker='o', label=f'{name} (μ={mean:.2f}%)',
                color=colour, linewidth=2, markersize=6)

        ax.fill_between(fold_numbers,
                        values - std,
                        values + std,
                        alpha=0.1, color=colour)

    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xlabel('Fold Number', fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(fold_numbers)

    # Set y axis limits based on metric
    all_vals = []
    for name in models.keys():
        all_vals.extend(cv_results[name][metric] * 100)
    min_val = max(0, min(all_vals) - 2)
    max_val = min(101, max(all_vals) + 2)
    ax.set_ylim(min_val, max_val)

fig.suptitle(
    '5-Fold Cross Validation Comparison\n'
    'Decision Tree vs Logistic Regression vs Random Forest',
    fontsize=14, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig("model_comparison_cv.png", dpi=150,
            bbox_inches='tight')
print("Chart saved as model_comparison_cv.png")
print("")
print("Cross validation comparison complete!")