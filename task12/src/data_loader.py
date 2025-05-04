import os
import re
import string
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Get English stopwords
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing punctuation, numbers, and stopwords.
    
    Args:
        text: Raw email text
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    
    # Join tokens back into a string
    return ' '.join(filtered_tokens)

def process_email_file(file_path: str) -> str:
    """
    Read and preprocess an email file.
    
    Args:
        file_path: Path to the email file
        
    Returns:
        Preprocessed email text
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return preprocess_text(content)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Return a minimal valid text instead of empty string to avoid empty dataset issues
        return "email content unavailable"

def process_emails_batch(files_batch: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """
    Process a batch of email files in parallel.
    
    Args:
        files_batch: List of tuples (file_path, label)
        
    Returns:
        List of tuples (preprocessed_text, label)
    """
    results = []
    for file_path, label in files_batch:
        preprocessed_text = process_email_file(file_path)
        results.append((preprocessed_text, label))
    return results

def load_enron_spam_dataset(data_dir: str = 'data/enron-spam', test_size: float = 0.2, 
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Load and preprocess the Enron-Spam dataset.
    
    Args:
        data_dir: Directory containing the dataset
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing preprocessed data and metadata
    """
    print("Loading Enron-Spam dataset...")
    
    # Get file paths and labels
    ham_dir = os.path.join(data_dir, 'ham')
    spam_dir = os.path.join(data_dir, 'spam')
    
    ham_files = [(os.path.join(ham_dir, f), 0) for f in os.listdir(ham_dir) if os.path.isfile(os.path.join(ham_dir, f))]
    spam_files = [(os.path.join(spam_dir, f), 1) for f in os.listdir(spam_dir) if os.path.isfile(os.path.join(spam_dir, f))]
    
    all_files = ham_files + spam_files
    
    print(f"Found {len(ham_files)} ham emails and {len(spam_files)} spam emails")
    
    # Process emails in parallel using multiprocessing
    num_cores = multiprocessing.cpu_count()
    batch_size = max(1, len(all_files) // (num_cores * 4))  # Adjust batch size based on number of cores
    batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
    
    print(f"Processing emails using {num_cores} cores with {len(batches)} batches...")
    
    processed_data = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for batch_result in tqdm(executor.map(process_emails_batch, batches), total=len(batches)):
            processed_data.extend(batch_result)
    
    # Create DataFrame
    emails_df = pd.DataFrame(processed_data, columns=['text', 'label'])
    
    # Remove empty emails
    emails_df = emails_df[emails_df['text'].str.strip() != '']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        emails_df['text'], 
        emails_df['label'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=emails_df['label']
    )
    
    print(f"Training set: {len(X_train)} emails")
    print(f"Test set: {len(X_test)} emails")
    
    # Create TF-IDF features
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit features to improve efficiency
        min_df=2,           # Ignore terms that appear in less than 2 documents
        max_df=0.9,         # Ignore terms that appear in more than 90% of documents
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF features shape: {X_train_tfidf.shape}")
    
    # Return preprocessed data and metadata
    return {
        'X_train': X_train_tfidf,
        'X_test': X_test_tfidf,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'vectorizer': vectorizer,
        'feature_names': vectorizer.get_feature_names_out(),
        'class_names': ['ham', 'spam']
    }

if __name__ == "__main__":
    # Test the data loader
    data = load_enron_spam_dataset()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of features: {len(data['feature_names'])}")
    print(f"Training set shape: {data['X_train'].shape}")
    print(f"Test set shape: {data['X_test'].shape}")
    print(f"Training set class distribution: {np.bincount(data['y_train'])}")
    print(f"Test set class distribution: {np.bincount(data['y_test'])}")
    
    # Print sample feature names
    print("\nSample features:")
    for i, feature in enumerate(data['feature_names'][:10]):
        print(f"  {i+1}. {feature}")