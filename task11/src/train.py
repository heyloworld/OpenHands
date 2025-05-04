import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple

from src.data_loader import load_and_preprocess_data
from src.model import HousePriceModel

def train_with_cross_validation(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_splits: int = 5, 
    random_state: int = 42
) -> Tuple[HousePriceModel, Dict[str, float], np.ndarray]:
    """
    Train the model using cross-validation.
    
    Args:
        X: Features
        y: Target values
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - Trained model
        - Dictionary of evaluation metrics
        - Cross-validation predictions
    """
    # Initialize model
    model = HousePriceModel()
    
    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize arrays to store results
    cv_mse = []
    cv_mae = []
    cv_r2 = []
    
    # Perform cross-validation
    print(f"Performing {n_splits}-fold cross-validation...")
    
    # Get cross-validation predictions
    cv_predictions = cross_val_predict(model.model, X, y, cv=kf)
    
    # Calculate metrics for each fold
    for train_idx, val_idx in kf.split(X):
        # Split data
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model on this fold
        fold_model = HousePriceModel()
        fold_model.train(X_train_fold, y_train_fold)
        
        # Evaluate on validation set
        y_val_pred = fold_model.predict(X_val_fold)
        
        # Calculate metrics
        mse = mean_squared_error(y_val_fold, y_val_pred)
        mae = mean_absolute_error(y_val_fold, y_val_pred)
        r2 = r2_score(y_val_fold, y_val_pred)
        
        # Store metrics
        cv_mse.append(mse)
        cv_mae.append(mae)
        cv_r2.append(r2)
    
    # Calculate average metrics
    avg_mse = np.mean(cv_mse)
    avg_rmse = np.sqrt(avg_mse)
    avg_mae = np.mean(cv_mae)
    avg_r2 = np.mean(cv_r2)
    
    # Print cross-validation results
    print("\nCross-Validation Results:")
    print(f"  MSE: {avg_mse:.4f} (±{np.std(cv_mse):.4f})")
    print(f"  RMSE: {avg_rmse:.4f}")
    print(f"  MAE: {avg_mae:.4f} (±{np.std(cv_mae):.4f})")
    print(f"  R²: {avg_r2:.4f} (±{np.std(cv_r2):.4f})")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    model.train(X, y)
    
    # Return final model, metrics, and CV predictions
    return model, {
        'mse': avg_mse,
        'rmse': avg_rmse,
        'mae': avg_mae,
        'r2': avg_r2,
        'cv_mse': cv_mse,
        'cv_mae': cv_mae,
        'cv_r2': cv_r2
    }, cv_predictions

def save_metrics(metrics: Dict[str, Any], file_path: str) -> None:
    """
    Save evaluation metrics to a file.
    
    Args:
        metrics: Dictionary of metrics
        file_path: Path to save the metrics
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Format metrics as string
    metrics_str = "House Price Prediction Metrics\n"
    metrics_str += "============================\n\n"
    
    # Add cross-validation metrics
    metrics_str += "Cross-Validation Metrics:\n"
    metrics_str += f"  Mean Squared Error (MSE): {metrics['mse']:.4f}\n"
    metrics_str += f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n"
    metrics_str += f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}\n"
    metrics_str += f"  R² Score: {metrics['r2']:.4f}\n\n"
    
    # Add test set metrics if available
    if 'test_mse' in metrics:
        metrics_str += "Test Set Metrics:\n"
        metrics_str += f"  Mean Squared Error (MSE): {metrics['test_mse']:.4f}\n"
        metrics_str += f"  Root Mean Squared Error (RMSE): {metrics['test_rmse']:.4f}\n"
        metrics_str += f"  Mean Absolute Error (MAE): {metrics['test_mae']:.4f}\n"
        metrics_str += f"  R² Score: {metrics['test_r2']:.4f}\n"
    
    # Write to file
    with open(file_path, 'w') as f:
        f.write(metrics_str)
    
    print(f"Metrics saved to {file_path}")

def visualize_predictions(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    title: str, 
    file_path: str
) -> None:
    """
    Visualize the comparison between predicted and actual values.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
        file_path: Path to save the visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot actual vs predicted values
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    
    # Add metrics to plot
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    plt.figtext(0.15, 0.8, f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {file_path}")

def main():
    """Main function to train and evaluate the model."""
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Train model with cross-validation
    model, cv_metrics, cv_predictions = train_with_cross_validation(
        data['X_train'], data['y_train'], n_splits=5
    )
    
    # Evaluate on test set
    test_metrics = model.evaluate(data['X_test'], data['y_test'])
    
    # Combine metrics
    all_metrics = {**cv_metrics}
    all_metrics.update({
        'test_mse': test_metrics['mse'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2']
    })
    
    # Save metrics
    save_metrics(all_metrics, 'results/metrics/metrics.txt')
    
    # Make predictions on test set
    test_predictions = model.predict(data['X_test'])
    
    # Visualize cross-validation predictions
    visualize_predictions(
        data['y_train'], 
        cv_predictions, 
        'Cross-Validation: Predicted vs Actual House Prices',
        'results/figures/cv_prediction_vs_actual.png'
    )
    
    # Visualize test predictions
    visualize_predictions(
        data['y_test'], 
        test_predictions, 
        'Test Set: Predicted vs Actual House Prices',
        'results/figures/prediction_vs_actual.png'
    )
    
    print("\nTraining and evaluation completed successfully!")

if __name__ == "__main__":
    main()