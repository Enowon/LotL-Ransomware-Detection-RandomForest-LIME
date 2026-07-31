import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# ----------------------------------------
# Step 1 - Load and prepare the data
# ----------------------------------------
print("Loading dataset...")
df = pd.read_csv("Ransomware_Data.csv")

# Convert labels to numbers
df['label'] = df['Ware Type'].map({'good': 0, 'ransom': 1})

# Separate features and labels
X = df.drop(columns=['Ware Type', 'label'])
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Dataset loaded and split successfully!")
print("")

# ----------------------------------------
# Step 2 - Build and train the model
# ----------------------------------------
print("Training the Random Forest model...")
print("This may take a minute or two - please wait...")
print("")

# Create the Random Forest model
# n_estimators=100 means 100 decision trees
# random_state=42 means results will be the same every time we run it
# n_jobs=-1 means use all your computer's processors to speed things up
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model on the training data
model.fit(X_train, y_train)

print("Model trained successfully!")
print("")

# ----------------------------------------
# Step 3 - Save the model and test data
# ----------------------------------------
# We save the model so we do not have to retrain it every time
# Think of it like saving a Word document so you do not lose your work
print("Saving the model...")

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the test data so we can use it in the next step
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Model saved as random_forest_model.pkl")
print("Test data saved as X_test.csv and y_test.csv")
print("")
print("All done! Ready for evaluation.")