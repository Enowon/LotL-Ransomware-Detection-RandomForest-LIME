import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import time

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
# Step 2 - Define hyperparameter grid
# ----------------------------------------
print("Setting up hyperparameter grid...")
print("Grid Search will try every combination of these settings:")
print("")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

# Print all combinations
total = (len(param_grid['n_estimators']) *
         len(param_grid['max_depth']) *
         len(param_grid['min_samples_split']))

print(f"  n_estimators:     {param_grid['n_estimators']}")
print(f"  max_depth:        {param_grid['max_depth']}")
print(f"  min_samples_split:{param_grid['min_samples_split']}")
print(f"  Total combinations: {total}")
print(f"  Each tested with 3-fold CV = {total * 3} model fits")
print("")

# ----------------------------------------
# Step 3 - Run Grid Search
# ----------------------------------------
print("Starting Grid Search — this may take 30-45 minutes...")
print("Please wait and do not close the terminal...")
print("")

start_time = time.time()

# Use F1 score as the optimisation metric
# because our dataset is imbalanced
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

end_time = time.time()
elapsed = (end_time - start_time) / 60
print("")
print(f"Grid Search completed in {elapsed:.1f} minutes!")
print("")

# ----------------------------------------
# Step 4 - Print best parameters
# ----------------------------------------
print("=" * 50)
print("BEST HYPERPARAMETERS FOUND")
print("=" * 50)
best_params = grid_search.best_params_
print(f"  n_estimators:      {best_params['n_estimators']}")
print(f"  max_depth:         {best_params['max_depth']}")
print(f"  min_samples_split: {best_params['min_samples_split']}")
print(f"  Best CV F1 Score:  {grid_search.best_score_*100:.2f}%")
print("=" * 50)
print("")

# ----------------------------------------
# Step 5 - Evaluate tuned model
# ----------------------------------------
print("Evaluating tuned model on test data...")
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

acc_tuned = accuracy_score(y_test, y_pred_tuned)
prec_tuned = precision_score(y_test, y_pred_tuned)
rec_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)

print("")
print("=" * 55)
print("COMPARISON: ORIGINAL vs TUNED MODEL")
print("=" * 55)
print(f"{'Metric':<12} {'Original':>12} {'Tuned':>12} {'Change':>10}")
print("-" * 55)

original = {
    'Accuracy': 98.15,
    'Precision': 93.04,
    'Recall': 99.37,
    'F1 Score': 96.10
}

tuned = {
    'Accuracy': acc_tuned*100,
    'Precision': prec_tuned*100,
    'Recall': rec_tuned*100,
    'F1 Score': f1_tuned*100
}

for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    orig_val = original[metric]
    tuned_val = tuned[metric]
    change = tuned_val - orig_val
    direction = "↑" if change > 0 else "↓" if change < 0 else "="
    print(f"{metric:<12} {orig_val:>11.2f}% {tuned_val:>11.2f}% "
          f"{direction}{abs(change):>8.2f}%")

print("=" * 55)
print("")

# ----------------------------------------
# Step 6 - Save the tuned model
# ----------------------------------------
print("Saving tuned model...")
with open("tuned_random_forest_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Tuned model saved as tuned_random_forest_model.pkl")
print("")

# ----------------------------------------
# Step 7 - Save comparison chart
# ----------------------------------------
print("Saving comparison chart...")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
original_scores = [original[m] for m in metrics]
tuned_scores = [tuned[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, original_scores, width,
               label='Original Model (Default Settings)',
               color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, tuned_scores, width,
               label='Tuned Model (Grid Search)',
               color='#2ecc71', alpha=0.8)

ax.set_ylim(88, 101)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Hyperparameter Tuning Results\nOriginal vs Tuned Random Forest Model',
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("hyperparameter_tuning.png", dpi=150, bbox_inches='tight')
print("Chart saved as hyperparameter_tuning.png")
print("")
print("Hyperparameter tuning complete!")