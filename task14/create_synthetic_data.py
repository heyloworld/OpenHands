"""
Script to create a synthetic Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic data
X, y = make_classification(
    n_samples=5000,
    n_features=15,
    n_informative=10,
    n_redundant=3,
    n_classes=2,
    weights=[0.7, 0.3],  # Imbalanced classes
    random_state=42
)

# Create a DataFrame
feature_names = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber',
    'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
    'TechSupport_Yes'
]

df = pd.DataFrame(X, columns=feature_names)

# Add customer ID
df['customerID'] = ['CUST-' + str(i).zfill(5) for i in range(len(df))]

# Add target variable (Churn)
df['Churn'] = y

# Normalize features to appropriate ranges
df['SeniorCitizen'] = (df['SeniorCitizen'] > 0.5).astype(int)
df['tenure'] = np.abs(df['tenure'] * 72).astype(int)  # 0-72 months
df['MonthlyCharges'] = 20 + np.abs(df['MonthlyCharges'] * 100)  # $20-$120
df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']

# Convert binary features to 0/1
binary_features = [
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_Yes', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
    'DeviceProtection_Yes', 'TechSupport_Yes'
]
for feature in binary_features:
    df[feature] = (df[feature] > 0.5).astype(int)

# Ensure InternetService is mutually exclusive
df['InternetService_DSL'] = (df['InternetService_DSL'] > 0.5).astype(int)
df['InternetService_Fiber'] = (df['InternetService_Fiber'] > 0.5).astype(int)
mask = (df['InternetService_DSL'] == 1) & (df['InternetService_Fiber'] == 1)
df.loc[mask, 'InternetService_Fiber'] = 0

# Add categorical contract type
contract_types = ['Month-to-month', 'One year', 'Two year']
df['Contract'] = np.random.choice(contract_types, size=len(df), p=[0.6, 0.3, 0.1])

# Add categorical payment method
payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
df['PaymentMethod'] = np.random.choice(payment_methods, size=len(df))

# Add paperless billing
df['PaperlessBilling'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])

# Convert binary features to Yes/No for better readability
for feature in binary_features + ['SeniorCitizen', 'PaperlessBilling']:
    if feature in df.columns:
        df[feature] = df[feature].map({0: 'No', 1: 'Yes'})

# Convert Churn to Yes/No
df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})

# Reorder columns to match the original dataset
df = df[[
    'customerID', 'gender_Male', 'SeniorCitizen', 'Partner_Yes', 'Dependents_Yes',
    'tenure', 'PhoneService_Yes', 'MultipleLines_Yes', 'InternetService_DSL',
    'InternetService_Fiber', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
    'DeviceProtection_Yes', 'TechSupport_Yes', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]]

# Rename columns to match the original dataset
df = df.rename(columns={
    'gender_Male': 'gender',
    'Partner_Yes': 'Partner',
    'Dependents_Yes': 'Dependents',
    'PhoneService_Yes': 'PhoneService',
    'MultipleLines_Yes': 'MultipleLines',
    'InternetService_DSL': 'InternetService_DSL',
    'InternetService_Fiber': 'InternetService_Fiber',
    'OnlineSecurity_Yes': 'OnlineSecurity',
    'OnlineBackup_Yes': 'OnlineBackup',
    'DeviceProtection_Yes': 'DeviceProtection',
    'TechSupport_Yes': 'TechSupport'
})

# Combine InternetService columns
df['InternetService'] = 'No'
df.loc[df['InternetService_DSL'] == 'Yes', 'InternetService'] = 'DSL'
df.loc[df['InternetService_Fiber'] == 'Yes', 'InternetService'] = 'Fiber optic'
df = df.drop(['InternetService_DSL', 'InternetService_Fiber'], axis=1)

# Map gender to Female/Male
df['gender'] = df['gender'].map({'No': 'Female', 'Yes': 'Male'})

# Save the synthetic dataset
df.to_csv('/workspace/OpenHands/data/churn_synthetic.csv', index=False)

print(f"Synthetic dataset created with {len(df)} samples")
print(f"Churn distribution: {df['Churn'].value_counts()}")
print(f"Dataset saved to /workspace/OpenHands/data/churn_synthetic.csv")