# Spam Email Detection

This project implements a spam email detection system using Support Vector Machine (SVM) classifier on the Enron-Spam dataset. The system preprocesses email text by removing stop words and punctuation, employs TF-IDF features, and performs hyperparameter tuning using GridSearchCV.

## Project Structure

```
.
├── data/
│   └── enron-spam/       # Dataset directory
│       ├── ham/          # Non-spam emails
│       └── spam/         # Spam emails
├── models/
│   └── saved_models/     # Saved model files
├── results/
│   ├── figures/          # Visualizations
│   └── classification_report.pdf  # Detailed report
├── src/
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── model.py          # SVM classifier implementation
│   └── train.py          # Training and evaluation
└── main.py               # Main script
```

## Features

- **Efficient Text Preprocessing**: Removes stop words, punctuation, and numbers from emails using parallel processing for improved performance.
- **TF-IDF Feature Extraction**: Converts text data into numerical features using Term Frequency-Inverse Document Frequency.
- **SVM Classification**: Implements Support Vector Machine classifier for spam detection.
- **Hyperparameter Tuning**: Uses GridSearchCV to find optimal model parameters.
- **Comprehensive Evaluation**: Calculates precision, recall, F1-score, and generates confusion matrix.
- **Detailed Reporting**: Creates a PDF report with all performance metrics and visualizations.

## Requirements

- Python 3.6+
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- nltk
- reportlab
- tqdm

## Usage

### Training the Model

To train the model with hyperparameter tuning:

```bash
python main.py --train --tune_params
```

To train the model without hyperparameter tuning:

```bash
python main.py --train
```

### Evaluating the Model

To evaluate a trained model:

```bash
python main.py --evaluate
```

### Making Predictions

To predict whether a single email is spam or not:

```bash
python main.py --predict path/to/email.txt
```

## Results

The model's performance is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The detailed results are saved in `results/classification_report.pdf`, and the confusion matrix visualization is saved in `results/figures/confusion_matrix.png`.

## Implementation Details

### Data Preprocessing

- Emails are converted to lowercase
- Punctuation and numbers are removed
- Stop words are filtered out
- Text is tokenized and normalized

### Feature Extraction

- TF-IDF vectorization with unigrams and bigrams
- Feature selection to limit to the most informative features
- Normalization of feature vectors

### Model Training

- SVM classifier with various kernel options
- Hyperparameter tuning for optimal performance
- Cross-validation to ensure robustness

## License

This project is licensed under the MIT License - see the LICENSE file for details.