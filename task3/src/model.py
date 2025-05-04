import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import time
from wordcloud import WordCloud
import seaborn as sns
from src.data_loader import load_data, generate_wordcloud, generate_class_wordclouds

class NewsGroupsClassifier:
    """
    Naive Bayes classifier for the 20 Newsgroups dataset.
    """
    
    def __init__(self, alpha=1.0, use_tfidf=True):
        """
        Initialize the classifier.
        
        Args:
            alpha (float): Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
            use_tfidf (bool): Whether to use TF-IDF features.
        """
        self.alpha = alpha
        self.use_tfidf = use_tfidf
        self.model = MultinomialNB(alpha=alpha)
        self.vectorizer = None
        self.classes = None
        self.feature_names = None
        self.feature_log_prob = None
    
    def train(self, X_train, y_train):
        """
        Train the classifier.
        
        Args:
            X_train: Training data features
            y_train: Training data labels
            
        Returns:
            self: The trained classifier
        """
        print(f"Training Multinomial Naive Bayes classifier with alpha={self.alpha}...")
        start_time = time.time()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Store model attributes for interpretation
        self.classes = self.model.classes_
        self.feature_log_prob = self.model.feature_log_prob_
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Data features
            
        Returns:
            array: Predicted labels
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, target_names=None, output_file=None):
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test data features
            y_test: Test data labels
            target_names (list): List of target names
            output_file (str): Path to save the evaluation report
            
        Returns:
            dict: Classification report as a dictionary
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=target_names, 
            output_dict=True
        )
        
        # Print the report
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Save the report to a file if specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(f"Naive Bayes Classifier Evaluation Report\n")
                f.write(f"=======================================\n\n")
                f.write(f"Model Parameters:\n")
                f.write(f"- Alpha (smoothing): {self.alpha}\n")
                f.write(f"- Feature type: {'TF-IDF' if self.use_tfidf else 'Count'}\n\n")
                f.write(f"Performance Metrics:\n")
                f.write(f"- Accuracy: {accuracy:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_test, y_pred, target_names=target_names))
                
                # Add timestamp
                f.write(f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"Evaluation report saved to {output_file}")
        
        return report
    
    def plot_confusion_matrix(self, X_test, y_test, target_names=None, filename=None):
        """
        Plot the confusion matrix.
        
        Args:
            X_test: Test data features
            y_test: Test data labels
            target_names (list): List of target names
            filename (str): Path to save the confusion matrix plot
            
        Returns:
            None
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=target_names if target_names else self.classes,
            yticklabels=target_names if target_names else self.classes
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
            print(f"Confusion matrix saved to {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def get_top_features(self, vectorizer, n_top=10):
        """
        Get the top features (words) for each class.
        
        Args:
            vectorizer: The vectorizer used to transform the data
            n_top (int): Number of top features to return
            
        Returns:
            dict: Dictionary mapping class names to lists of top features
        """
        if not hasattr(self.model, 'feature_log_prob_'):
            raise ValueError("Model has not been trained yet")
        
        feature_names = np.array(vectorizer.get_feature_names_out())
        top_features = {}
        
        for i, c in enumerate(self.model.classes_):
            # Get the feature log probabilities for this class
            feature_probs = self.model.feature_log_prob_[i]
            
            # Get the indices of the top features
            top_indices = feature_probs.argsort()[-n_top:][::-1]
            
            # Get the feature names
            top_features[c] = feature_names[top_indices]
        
        return top_features
    
    def visualize_top_features(self, vectorizer, target_names=None, n_top=20, filename=None):
        """
        Visualize the top features for each class.
        
        Args:
            vectorizer: The vectorizer used to transform the data
            target_names (list): List of target names
            n_top (int): Number of top features to visualize
            filename (str): Path to save the visualization
            
        Returns:
            None
        """
        top_features = self.get_top_features(vectorizer, n_top=n_top)
        
        # Create a figure with subplots
        n_classes = len(top_features)
        n_cols = 2
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()
        
        for i, (c, features) in enumerate(top_features.items()):
            # Get the feature log probabilities for this class
            feature_probs = np.exp(self.model.feature_log_prob_[c])
            
            # Get the indices of the top features
            top_indices = feature_probs.argsort()[-n_top:][::-1]
            
            # Get the feature names and probabilities
            top_features_names = np.array(vectorizer.get_feature_names_out())[top_indices]
            top_features_probs = feature_probs[top_indices]
            
            # Plot the top features
            axes[i].barh(range(n_top), top_features_probs, align='center')
            axes[i].set_yticks(range(n_top))
            axes[i].set_yticklabels(top_features_names)
            axes[i].invert_yaxis()
            axes[i].set_title(f"Class {target_names[c] if target_names else c}")
            axes[i].set_xlabel('Probability')
        
        # Hide any unused subplots
        for i in range(n_classes, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
            print(f"Top features visualization saved to {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_wordcloud_for_class(self, class_idx, vectorizer, filename=None):
        """
        Generate a word cloud for a specific class based on feature importance.
        
        Args:
            class_idx (int): Index of the class
            vectorizer: The vectorizer used to transform the data
            filename (str): Path to save the word cloud image
            
        Returns:
            None
        """
        if not hasattr(self.model, 'feature_log_prob_'):
            raise ValueError("Model has not been trained yet")
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature log probabilities for this class
        feature_probs = self.model.feature_log_prob_[class_idx]
        
        # Convert log probabilities to probabilities
        feature_probs = np.exp(feature_probs)
        
        # Create a dictionary of word frequencies
        word_freq = {feature_names[i]: feature_probs[i] for i in range(len(feature_names))}
        
        # Create the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate_from_frequencies(word_freq)
        
        # Display the word cloud
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for Class {class_idx}")
        
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
            print(f"Word cloud saved to {filename}")
        else:
            plt.show()
        
        plt.close()

def train_and_evaluate(categories=None, alpha=1.0, use_tfidf=True, test_size=0.2, random_state=42):
    """
    Train and evaluate a Naive Bayes classifier on the 20 Newsgroups dataset.
    
    Args:
        categories (list): List of categories to use. If None, all categories are used.
        alpha (float): Smoothing parameter for Naive Bayes.
        use_tfidf (bool): Whether to use TF-IDF features.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        
    Returns:
        tuple: (classifier, vectorizer, raw_data)
    """
    # Load the data
    X_train, X_test, y_train, y_test, vectorizer, raw_data = load_data(
        categories=categories,
        remove_stopwords=True,
        use_tfidf=use_tfidf,
        test_size=test_size,
        random_state=random_state
    )
    
    # Generate word cloud before training
    generate_wordcloud(
        raw_data, 
        vectorizer, 
        filename="results/figures/wordcloud_before.png"
    )
    
    # Create and train the classifier
    classifier = NewsGroupsClassifier(alpha=alpha, use_tfidf=use_tfidf)
    classifier.train(X_train, y_train)
    
    # Evaluate the classifier
    classifier.evaluate(
        X_test, 
        y_test, 
        target_names=raw_data.target_names,
        output_file="results/metrics/performance.txt"
    )
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(
        X_test, 
        y_test, 
        target_names=raw_data.target_names,
        filename="results/figures/confusion_matrix.png"
    )
    
    # Generate word cloud after training for a few classes
    for i in range(min(5, len(raw_data.target_names))):
        classifier.generate_wordcloud_for_class(
            i, 
            vectorizer, 
            filename=f"results/figures/wordcloud_class_{i}.png"
        )
    
    # Generate combined word cloud after training
    combined_wordcloud_data = {}
    for i, class_name in enumerate(raw_data.target_names):
        # Get feature log probabilities for this class
        feature_probs = classifier.model.feature_log_prob_[i]
        
        # Convert log probabilities to probabilities
        feature_probs = np.exp(feature_probs)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Add to combined data with class weight
        for j, name in enumerate(feature_names):
            if name in combined_wordcloud_data:
                combined_wordcloud_data[name] += feature_probs[j]
            else:
                combined_wordcloud_data[name] = feature_probs[j]
    
    # Create the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=200,
        contour_width=3,
        contour_color='steelblue'
    ).generate_from_frequencies(combined_wordcloud_data)
    
    # Display the word cloud
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud After Training")
    plt.savefig("results/figures/wordcloud_after.png", bbox_inches='tight')
    plt.close()
    
    print("Word cloud after training saved to results/figures/wordcloud_after.png")
    
    return classifier, vectorizer, raw_data

if __name__ == "__main__":
    # Train and evaluate the classifier
    classifier, vectorizer, raw_data = train_and_evaluate(alpha=1.0, use_tfidf=True)