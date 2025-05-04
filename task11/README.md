# House Price Prediction Project

This project implements a Linear Regression model to predict house prices using the California Housing dataset. The model is trained with cross-validation and evaluated using various metrics.

## Project Structure

```
.
├── src/
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── model.py        # Linear Regression model implementation
│   └── train.py        # Training and evaluation with cross-validation
├── results/
│   ├── metrics/        # Performance metrics
│   └── figures/        # Visualizations
└── main.py             # Main script to run the pipeline
```

## Features

- **Data Loading and Preprocessing**: Loads the California Housing dataset and performs feature scaling and standardization.
- **Linear Regression Model**: Implements a simple yet effective Linear Regression model for house price prediction.
- **Cross-Validation**: Uses 5-fold cross-validation to evaluate the model's performance.
- **Performance Metrics**: Calculates and reports Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score.
- **Visualizations**: Creates scatter plots comparing predicted vs. actual house prices.

## Usage

To train the model and generate metrics and visualizations:

```bash
python main.py --train
```

## Results

The model achieves the following performance metrics:

### Cross-Validation Metrics
- Mean Squared Error (MSE): 0.5193
- Root Mean Squared Error (RMSE): 0.7206
- Mean Absolute Error (MAE): 0.5291
- R² Score: 0.6115

### Test Set Metrics
- Mean Squared Error (MSE): 0.5559
- Root Mean Squared Error (RMSE): 0.7456
- Mean Absolute Error (MAE): 0.5332
- R² Score: 0.5758

## Feature Importance

The model identifies the following features as most important (based on coefficient magnitude):
1. Latitude (-0.8969)
2. Longitude (-0.8698)
3. Median Income (0.8544)
4. Average Bedrooms (0.3393)
5. Average Rooms (-0.2944)

## Visualizations

The project generates two visualizations:
1. `results/figures/cv_prediction_vs_actual.png`: Scatter plot of predicted vs. actual values from cross-validation
2. `results/figures/prediction_vs_actual.png`: Scatter plot of predicted vs. actual values on the test set

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- datasets (Hugging Face)