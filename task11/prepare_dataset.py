import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import warnings
from datasets import Dataset, DatasetDict

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directory if it doesn't exist
os.makedirs('mrseba', exist_ok=True)

# Load the California Housing dataset (as a replacement for Boston)
print("Loading California Housing dataset")
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['PRICE'] = california.target

# Rename columns to match Boston dataset style (uppercase)
data.columns = [col.upper() for col in data.columns]

print("Dataset loaded with features:")
for i, feature in enumerate(california.feature_names):
    print(f"  {feature.upper()}: {california.DESCR.split('- ')[i+1].split(':')[0]}")

# Split into train and test sets (80% train, 20% test)
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Save the dataset
dataset_dict.save_to_disk('mrseba/boston_house_price')

print(f"Dataset saved to mrseba/boston_house_price")
print(f"Dataset shape: {data.shape}")
print(f"Features: {', '.join(data.columns[:-1])}")
print(f"Target: PRICE")
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")