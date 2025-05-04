"""
Script to fix the Telco Customer Churn dataset.
"""

import pandas as pd

# Load the data
data_path = '/workspace/OpenHands/data/churn.csv'
df = pd.read_csv(data_path)

# Print the shape and columns
print(f"Original data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Convert 'TotalCharges' to numeric, handling any errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Replace any NaN values in TotalCharges with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Save the fixed data
fixed_data_path = '/workspace/OpenHands/data/churn_fixed.csv'
df.to_csv(fixed_data_path, index=False)

print(f"Fixed data saved to {fixed_data_path}")
print(f"Fixed data shape: {df.shape}")