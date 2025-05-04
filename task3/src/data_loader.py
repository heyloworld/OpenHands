import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import re
import os
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import stopwords

def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text by removing special characters, numbers, and converting to lowercase.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove headers, footers, and quotes (specific to 20 Newsgroups dataset)
    text = re.sub(r'From:.*\n', ' ', text)
    text = re.sub(r'Subject:.*\n', ' ', text)
    text = re.sub(r'[^\s]*@[^\s]*', ' ', text)  # Remove email addresses
    text = re.sub(r'[^\s]*http[^\s]*', ' ', text)  # Remove URLs
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_data(categories=None, remove_stopwords=True, use_tfidf=True, test_size=0.2, random_state=42):
    """
    Load and preprocess the 20 Newsgroups dataset.
    
    Args:
        categories (list): List of categories to load. If None, all categories are loaded.
        remove_stopwords (bool): Whether to remove stopwords.
        use_tfidf (bool): Whether to use TF-IDF features.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer, raw_data)
    """
    # Download NLTK resources if needed
    download_nltk_resources()
    
    # Load the dataset
    print("Loading 20 Newsgroups dataset...")
    if categories:
        print(f"Selected categories: {categories}")
    else:
        print("Using all categories")
    
    # Load training data
    train_data = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=random_state,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Load test data
    test_data = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=random_state,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Get the raw data for visualization
    raw_train_data = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=random_state,
        remove=()  # Keep everything for raw data
    )
    
    # Preprocess the data
    print("Preprocessing data...")
    preprocessed_train_texts = [preprocess_text(text) for text in train_data.data]
    preprocessed_test_texts = [preprocess_text(text) for text in test_data.data]
    
    # Get stop words
    stop_words = 'english' if remove_stopwords else None
    
    # Create the vectorizer
    if use_tfidf:
        print("Using TF-IDF vectorization...")
        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            min_df=5,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            sublinear_tf=True  # Apply sublinear tf scaling (log(tf))
        )
    else:
        print("Using Count vectorization...")
        vectorizer = CountVectorizer(
            stop_words=stop_words,
            min_df=5,
            max_df=0.8
        )
    
    # Transform the data
    X_train = vectorizer.fit_transform(preprocessed_train_texts)
    X_test = vectorizer.transform(preprocessed_test_texts)
    
    y_train = train_data.target
    y_test = test_data.target
    
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, vectorizer, raw_train_data

def generate_wordcloud(data, vectorizer, top_n_classes=None, filename=None):
    """
    Generate a word cloud from the data.
    
    Args:
        data: The raw data object from fetch_20newsgroups
        vectorizer: The vectorizer used to transform the data
        top_n_classes (int): Number of top classes to include. If None, all classes are included.
        filename (str): Path to save the word cloud image. If None, the image is displayed.
        
    Returns:
        None
    """
    # Combine all texts
    all_text = ' '.join([preprocess_text(text) for text in data.data])
    
    # Create the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=200,
        contour_width=3,
        contour_color='steelblue'
    ).generate(all_text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if filename:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')
        print(f"Word cloud saved to {filename}")
    else:
        plt.show()
    
    plt.close()

def generate_class_wordclouds(data, vectorizer, class_names, filename_prefix=None):
    """
    Generate word clouds for each class.
    
    Args:
        data: The raw data object from fetch_20newsgroups
        vectorizer: The vectorizer used to transform the data
        class_names (list): List of class names
        filename_prefix (str): Prefix for the saved files. If None, the images are displayed.
        
    Returns:
        None
    """
    # Create a figure with subplots
    n_classes = len(class_names)
    n_cols = 2
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        # Get texts for this class
        class_indices = [j for j, target in enumerate(data.target) if data.target_names[target] == class_name]
        class_texts = [data.data[j] for j in class_indices]
        
        if not class_texts:
            continue
        
        # Combine all texts for this class
        class_text = ' '.join([preprocess_text(text) for text in class_texts])
        
        # Create the word cloud
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color='white',
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        ).generate(class_text)
        
        # Display the word cloud
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(class_name)
        axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if filename_prefix:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)
        plt.savefig(f"{filename_prefix}_classes.png", bbox_inches='tight')
        print(f"Class word clouds saved to {filename_prefix}_classes.png")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Test the data loader
    X_train, X_test, y_train, y_test, vectorizer, raw_data = load_data()
    
    # Generate word cloud
    generate_wordcloud(raw_data, vectorizer, filename="results/figures/wordcloud_test.png")
    
    # Generate class word clouds for a few classes
    selected_classes = raw_data.target_names[:5]  # First 5 classes
    generate_class_wordclouds(raw_data, vectorizer, selected_classes, filename_prefix="results/figures/wordcloud_test")