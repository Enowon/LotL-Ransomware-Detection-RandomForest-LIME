import pandas as pd
import pickle
import numpy as np
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# Step 2 - Set up LIME explainer
# ----------------------------------------
print("Setting up LIME explainer...")

# LIME needs to know the feature names and that this is a classification task
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_test.values,
    feature_names=X_test.columns.tolist(),
    class_names=['Benign', 'Ransomware'],
    mode='classification'
)

print("LIME explainer ready!")
print("")

# ----------------------------------------
# Step 3 - Select entries to explain
# ----------------------------------------
# Get model predictions on test data
y_pred = model.predict(X_test)
y_test_array = y_test.values

# Find indices for each category
# Correctly identified ransomware (True Positives)
tp_indices = np.where((y_pred == 1) & (y_test_array == 1))[0]

# Correctly identified benign (True Negatives)
tn_indices = np.where((y_pred == 0) & (y_test_array == 0))[0]

# Misclassified - ransomware missed by model (False Negatives)
fn_indices = np.where((y_pred == 0) & (y_test_array == 1))[0]

print(f"True Positives available:  {len(tp_indices)}")
print(f"True Negatives available:  {len(tn_indices)}")
print(f"False Negatives available: {len(fn_indices)}")
print("")

# ----------------------------------------
# Step 4 - Generate and save LIME explanations
# ----------------------------------------

def explain_and_save(index, label, filename):
    print(f"Generating explanation for: {label}")
    instance = X_test.values[index]
    actual = "Ransomware" if y_test_array[index] == 1 else "Benign"
    predicted = "Ransomware" if y_pred[index] == 1 else "Benign"

    print(f"  Actual label:    {actual}")
    print(f"  Predicted label: {predicted}")

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=10
    )

    # Print top features to terminal
    print(f"  Top features driving this prediction:")
    for feature, weight in explanation.as_list():
        direction = "towards Ransomware" if weight > 0 else "towards Benign"
        print(f"    {feature}: {weight:.4f} ({direction})")
    print("")

    # Save explanation as image
    fig = explanation.as_pyplot_figure()
    plt.title(f"LIME Explanation - {label}\nActual: {actual} | Predicted: {predicted}")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved as {filename}")
    print("")

# Explain 2 true positives (correctly caught ransomware)
explain_and_save(tp_indices[0], "Ransomware Case 1 (Correctly Detected)",
                 "lime_ransomware_1.png")
explain_and_save(tp_indices[1], "Ransomware Case 2 (Correctly Detected)",
                 "lime_ransomware_2.png")

# Explain 2 true negatives (correctly identified benign)
explain_and_save(tn_indices[0], "Benign Case 1 (Correctly Identified)",
                 "lime_benign_1.png")
explain_and_save(tn_indices[1], "Benign Case 2 (Correctly Identified)",
                 "lime_benign_2.png")

# Explain 1 false negative (ransomware the model missed)
explain_and_save(fn_indices[0], "Missed Ransomware Case (False Negative)",
                 "lime_false_negative.png")

# ----------------------------------------
# Step 5 - Summary
# ----------------------------------------
print("=" * 40)
print("LIME EXPLANATION SUMMARY")
print("=" * 40)
print("Five explanation images have been saved:")
print("  lime_ransomware_1.png  - Correctly detected ransomware")
print("  lime_ransomware_2.png  - Correctly detected ransomware")
print("  lime_benign_1.png      - Correctly identified benign")
print("  lime_benign_2.png      - Correctly identified benign")
print("  lime_false_negative.png - Ransomware the model missed")
print("")
print("These images can be included directly in Chapter 4")
print("of your thesis report.")
print("=" * 40)