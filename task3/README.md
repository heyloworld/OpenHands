# Naive Bayes Classifier for 20 Newsgroups Dataset

This project implements a Naive Bayes classifier for the 20 Newsgroups dataset. The classifier is trained to categorize text documents into one of 20 different newsgroup categories.

## Project Structure

```
.
├── main.py                             # Main script to run the classifier
├── src/
│   ├── data_loader.py                  # Data loading and preprocessing
│   └── model.py                        # Naive Bayes classifier implementation
├── results/
│   ├── figures/                        # Visualizations
│   │   ├── wordcloud_before.png        # Word cloud before training
│   │   ├── wordcloud_after.png         # Word cloud after training
│   │   └── confusion_matrix.png        # Confusion matrix
│   └── metrics/                        # Performance metrics
│       └── performance.txt             # Classification report
└── README.md                           # Project documentation
```

## Requirements

- Python 3.6+
- scikit-learn
- numpy
- matplotlib
- wordcloud
- seaborn
- nltk

## Features

- **Data Preprocessing**: Removes stop words, punctuation, and special characters
- **TF-IDF Features**: Uses TF-IDF features for better classification
- **Visualization**: Generates word clouds and confusion matrix
- **Performance Metrics**: Calculates precision, recall, and F1-score

## Usage

To train and evaluate the classifier with default parameters:

```bash
python main.py
```

### Command Line Arguments

- `--categories`: Comma-separated list of categories to use (default: all categories)
- `--alpha`: Smoothing parameter for Naive Bayes (default: 1.0)
- `--use-tfidf`: Use TF-IDF features instead of raw counts (default: True)
- `--test-size`: Proportion of the dataset to include in the test split (default: 0.2)
- `--random-state`: Random state for reproducibility (default: 42)

### Examples

Train on a subset of categories:

```bash
python main.py --categories alt.atheism,comp.graphics,sci.med,soc.religion.christian
```

Adjust the smoothing parameter:

```bash
python main.py --alpha 0.5
```

## Results

The classifier generates the following outputs:

1. **Word Clouds**:
   - `wordcloud_before.png`: Word cloud of the raw data
   - `wordcloud_after.png`: Word cloud based on feature importance after training

2. **Performance Metrics**:
   - `performance.txt`: Classification report with precision, recall, and F1-score

3. **Confusion Matrix**:
   - `confusion_matrix.png`: Visualization of the classifier's predictions

## How It Works

### Data Preprocessing

The data preprocessing pipeline includes:
- Converting text to lowercase
- Removing headers, footers, and quotes
- Removing email addresses and URLs
- Removing punctuation and numbers
- Removing stop words

### TF-IDF Features

The classifier uses Term Frequency-Inverse Document Frequency (TF-IDF) features, which:
- Gives higher weight to terms that are frequent in a document but rare across the corpus
- Reduces the impact of common words that appear in many documents
- Improves classification performance compared to raw word counts

### Naive Bayes Classifier

The Multinomial Naive Bayes classifier:
- Assumes features are generated from a multinomial distribution
- Uses Bayes' theorem to calculate the probability of a document belonging to a class
- Applies Laplace smoothing to handle unseen words

## Interpretation

The word clouds and top features for each class provide insights into:
- The most important words for each category
- How the classifier distinguishes between different categories
- The impact of preprocessing and feature selection

## License

This project is licensed under the MIT License - see the LICENSE file for details.