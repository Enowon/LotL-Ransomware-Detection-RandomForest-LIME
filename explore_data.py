import pandas as pd

# Load the dataset
df = pd.read_csv("Ransomware_Data.csv")

# How many rows and columns are there?
print("Dataset shape:")
print(df.shape)
print("")

# What are the column names?
print("Column names:")
print(df.columns.tolist())
print("")

# What do the first 5 rows look like?
print("First 5 rows:")
print(df.head())
print("")

# How many benign and ransomware entries are there?
print("Label counts:")
print(df['Ware Type'].value_counts())