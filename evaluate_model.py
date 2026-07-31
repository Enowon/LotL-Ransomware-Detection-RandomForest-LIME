import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ----------------------------------------
# Step 1 - Load the saved model and test data
# ----------------------------------------
print("Loading model and test data...")

with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

print("Model and test data loaded successfully!")
print("")

# ----------------------------------------
# Step 2 - Make predictions
# ----------------------------------------
print("Making predictions on test data...")
y_pred = model.predict(X_test)
print("Predictions complete!")
print("")

# ----------------------------------------
# Step 3 - Calculate performance metrics
# ----------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 40)
print("MODEL PERFORMANCE RESULTS")
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
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("")

# Explain the confusion matrix in plain terms
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
print(classification_report(y_test, y_pred,
      target_names=['Benign', 'Ransomware']))

# ----------------------------------------
# Step 6 - Save confusion matrix as image
# ----------------------------------------
print("Saving confusion matrix chart...")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im)

ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=['Benign', 'Ransomware'],
    yticklabels=['Benign', 'Ransomware'],
    ylabel='Actual Label',
    xlabel='Predicted Label',
    title='Random Forest Confusion Matrix'
)

for i in range(2):
    for j in range(2):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")
print("")
print("Evaluation complete!")