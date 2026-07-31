import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("Ransomware_Data.csv")

# Step 1 — Convert text labels to numbers
# "good" becomes 0, "ransom" becomes 1
df['label'] = df['Ware Type'].map({'good': 0, 'ransom': 1})

# Check the conversion worked
print("Label conversion check:")
print(df['label'].value_counts())
print("")

# Step 2 — Separate features (X) and labels (y)
# X is everything except the Ware Type and label columns
# y is just the label column
X = df.drop(columns=['Ware Type', 'label'])
y = df['label']

print("Features shape (X):")
print(X.shape)
print("")

print("Labels shape (y):")
print(y.shape)
print("")

# Step 3 — Split into training and testing sets
# 80% training, 20% testing
# random_state=42 just means the split will be the same every time we run it
# stratify=y makes sure both splits have the same proportion of good and ransom
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:")
print(X_train.shape)
print("")

print("Testing set size:")
print(X_test.shape)
print("")

print("Training label distribution:")
print(y_train.value_counts())
print("")

print("Testing label distribution:")
print(y_test.value_counts())
print("")

print("Preprocessing complete!")