# Sentiment Analysis on Tweets

This project implements a sentiment analysis system for tweets using Word2Vec for text vectorization and an SVM classifier for sentiment classification.

## Project Structure

```
.
├── src/
│   ├── data_loader.py         # Data loading and preprocessing (full implementation)
│   ├── simple_data_loader.py  # Simplified data loader for demonstration
│   └── model.py               # SVM classifier implementation
├── models/
│   └── saved_models/          # Directory for saved models
│       ├── word2vec_model.bin # Trained Word2Vec model
│       └── svm_classifier.pkl # Trained SVM classifier
├── results/
│   ├── figures/               # Directory for figures
│   │   └── confusion_matrix.png  # Confusion matrix visualization
│   └── metrics/               # Directory for metrics
│       └── accuracy_score.txt    # Evaluation metrics
├── main.py                    # Main script to run the sentiment analysis (full implementation)
├── simple_main.py             # Simplified main script for demonstration
├── test_sentiment.py          # Script to test the trained model
└── README.md                  # This file
```

## Requirements

- Python 3.6+
- scikit-learn
- nltk
- gensim
- pandas
- numpy
- matplotlib
- seaborn
- tqdm

## Implementation Details

### Data Loading and Preprocessing

The `DataLoader` class in `src/data_loader.py` is responsible for loading the dataset, preprocessing the text, and vectorizing it using Word2Vec. The preprocessing steps include:

1. Converting text to lowercase
2. Removing URLs, mentions, and hashtags
3. Removing special characters and numbers
4. Tokenizing the text
5. Removing stop words and lemmatizing the tokens

### Text Vectorization

The system uses Word2Vec for text vectorization. The Word2Vec model is trained on the cleaned tokens from the dataset. Each tweet is represented as the average of its word vectors.

### Model Training and Evaluation

The `SentimentClassifier` class in `src/model.py` implements an SVM classifier for sentiment classification. The class provides methods for training, evaluating, and making predictions with the classifier. The evaluation metrics include accuracy, precision, recall, F1 score, and a confusion matrix.

## Usage

### Training

To train the sentiment analysis model, run:

```bash
python simple_main.py [--sample_size SAMPLE_SIZE] [--test_size TEST_SIZE] [--random_state RANDOM_STATE] [--vector_size VECTOR_SIZE] [--window WINDOW] [--min_count MIN_COUNT] [--kernel KERNEL] [--C C]
```

Arguments:
- `--sample_size`: Number of samples to use. Default: 1000
- `--test_size`: Proportion of the dataset to include in the test split. Default: 0.2
- `--random_state`: Random state for reproducibility. Default: 42
- `--vector_size`: Dimensionality of the word vectors. Default: 100
- `--window`: Maximum distance between the current and predicted word. Default: 5
- `--min_count`: Ignores all words with total frequency lower than this. Default: 1
- `--kernel`: Kernel type to be used in the SVM algorithm. Choices: ["linear", "poly", "rbf", "sigmoid"]. Default: "linear"
- `--C`: Regularization parameter. Default: 1.0

### Testing

To test the trained model on custom tweets, run:

```bash
python test_sentiment.py [--word2vec_path WORD2VEC_PATH] [--model_path MODEL_PATH] [--tweets TWEETS [TWEETS ...]]
```

Arguments:
- `--word2vec_path`: Path to the trained Word2Vec model. Default: "models/saved_models/word2vec_model.bin"
- `--model_path`: Path to the trained SVM classifier. Default: "models/saved_models/svm_classifier.pkl"
- `--tweets`: List of tweets to test. Default: ["I love this product! It's amazing!", "This is the worst experience I've ever had.", "The weather is nice today.", "I'm feeling neutral about this situation."]

## Results

The evaluation results are saved to `results/metrics/accuracy_score.txt` and include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

A visualization of the confusion matrix is saved to `results/figures/confusion_matrix.png`.

## Note

This implementation uses a simplified dataset for demonstration purposes. For a full implementation with the Sentiment140 dataset from Hugging Face, use the `data_loader.py` and `main.py` scripts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.