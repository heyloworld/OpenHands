import os
import re
import logging
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, sample_size=None, test_size=0.2, random_state=42, vector_size=100, window=5, min_count=1):
        """
        Initialize the DataLoader.
        
        Args:
            sample_size (int, optional): Number of samples to use. If None, use all data.
            test_size (float, optional): Proportion of the dataset to include in the test split.
            random_state (int, optional): Random state for reproducibility.
            vector_size (int, optional): Dimensionality of the word vectors.
            window (int, optional): Maximum distance between the current and predicted word.
            min_count (int, optional): Ignores all words with total frequency lower than this.
        """
        self.sample_size = sample_size
        self.test_size = test_size
        self.random_state = random_state
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_model = None
        
    def load_data(self):
        """
        Load a small sample dataset for demonstration purposes.
        
        Returns:
            pandas.DataFrame: The loaded dataset.
        """
        logger.info("Loading sample dataset...")
        
        # Create a small sample dataset
        data = {
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 100,  # 0 = negative, 1 = positive
            'text': [
                "This is the worst product I've ever used. Terrible experience!",
                "I love this app! It's amazing and so helpful.",
                "The customer service was awful. They didn't help at all.",
                "Great experience with this company. Highly recommend!",
                "Disappointed with the quality. Would not buy again.",
                "Excellent service and fast delivery. Very satisfied!",
                "This movie was boring and a waste of time.",
                "The food was delicious and the staff was friendly.",
                "Terrible performance. Not worth the money.",
                "Best purchase I've made this year! So happy with it."
            ] * 100
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Take a sample if specified
        if self.sample_size is not None and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=self.random_state)
        
        logger.info(f"Sample dataset loaded with {len(df)} samples.")
        return df
    
    def clean_text(self, text):
        """
        Clean the text by removing URLs, mentions, hashtags, special characters, and stop words.
        
        Args:
            text (str): The text to clean.
            
        Returns:
            list: A list of cleaned tokens.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return cleaned_tokens
    
    def preprocess_data(self, df):
        """
        Preprocess the data by cleaning the text and tokenizing.
        
        Args:
            df (pandas.DataFrame): The dataset to preprocess.
            
        Returns:
            pandas.DataFrame: The preprocessed dataset.
        """
        logger.info("Preprocessing data...")
        
        # Clean the text
        tqdm.pandas(desc="Cleaning text")
        df['cleaned_tokens'] = df['text'].progress_apply(self.clean_text)
        
        # Remove empty token lists
        df = df[df['cleaned_tokens'].map(len) > 0]
        
        logger.info(f"Data preprocessing completed. {len(df)} samples remaining after cleaning.")
        return df
    
    def train_word2vec(self, df):
        """
        Train a Word2Vec model on the cleaned tokens.
        
        Args:
            df (pandas.DataFrame): The preprocessed dataset.
            
        Returns:
            gensim.models.Word2Vec: The trained Word2Vec model.
        """
        logger.info("Training Word2Vec model...")
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=df['cleaned_tokens'].tolist(),
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        
        logger.info(f"Word2Vec model trained with vocabulary size: {len(self.word2vec_model.wv.key_to_index)}")
        return self.word2vec_model
    
    def get_sentence_vector(self, tokens):
        """
        Get the vector representation of a sentence by averaging the word vectors.
        
        Args:
            tokens (list): A list of tokens.
            
        Returns:
            numpy.ndarray: The sentence vector.
        """
        # Get word vectors for each token
        word_vectors = [self.word2vec_model.wv[token] for token in tokens if token in self.word2vec_model.wv]
        
        # If no word vectors are found, return a zero vector
        if not word_vectors:
            return np.zeros(self.vector_size)
        
        # Average the word vectors
        return np.mean(word_vectors, axis=0)
    
    def vectorize_data(self, df):
        """
        Vectorize the data using the trained Word2Vec model.
        
        Args:
            df (pandas.DataFrame): The preprocessed dataset.
            
        Returns:
            tuple: X (features), y (labels), and the vectorized DataFrame.
        """
        logger.info("Vectorizing data...")
        
        # Vectorize the text
        tqdm.pandas(desc="Vectorizing text")
        df['vector'] = df['cleaned_tokens'].progress_apply(self.get_sentence_vector)
        
        # Convert vectors to a list of arrays
        X = np.array(df['vector'].tolist())
        y = df['target'].values
        
        logger.info(f"Data vectorization completed. X shape: {X.shape}, y shape: {y.shape}")
        return X, y, df
    
    def prepare_data(self):
        """
        Prepare the data for training by loading, preprocessing, and vectorizing.
        
        Returns:
            tuple: X_train, X_test, y_train, y_test, and the vectorized DataFrame.
        """
        # Load data
        df = self.load_data()
        
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Train Word2Vec model
        self.train_word2vec(df)
        
        # Vectorize data
        X, y, df = self.vectorize_data(df)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Data preparation completed. Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test, df
    
    def save_word2vec_model(self, path):
        """
        Save the trained Word2Vec model.
        
        Args:
            path (str): The path to save the model.
        """
        if self.word2vec_model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            self.word2vec_model.save(path)
            logger.info(f"Word2Vec model saved to {path}")
        else:
            logger.warning("No Word2Vec model to save.")
    
    def load_word2vec_model(self, path):
        """
        Load a trained Word2Vec model.
        
        Args:
            path (str): The path to load the model from.
            
        Returns:
            gensim.models.Word2Vec: The loaded Word2Vec model.
        """
        try:
            self.word2vec_model = Word2Vec.load(path)
            logger.info(f"Word2Vec model loaded from {path}")
            return self.word2vec_model
        except Exception as e:
            logger.error(f"Error loading Word2Vec model: {e}")
            raise

if __name__ == "__main__":
    # Test the DataLoader
    data_loader = DataLoader(sample_size=1000)
    X_train, X_test, y_train, y_test, df = data_loader.prepare_data()
    
    # Save the Word2Vec model
    data_loader.save_word2vec_model("models/word2vec_model.bin")
    
    # Print some statistics
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Positive samples in train set: {np.sum(y_train == 1)}")
    print(f"Negative samples in train set: {np.sum(y_train == 0)}")
    print(f"Positive samples in test set: {np.sum(y_test == 1)}")
    print(f"Negative samples in test set: {np.sum(y_test == 0)}")