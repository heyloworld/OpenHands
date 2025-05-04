import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple

class HousePriceModel:
    """
    Linear Regression model for house price prediction.
    """
    
    def __init__(self):
        """Initialize the model."""
        self.model = LinearRegression()
        self.is_trained = False
        self.feature_importance = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on the given data.
        
        Args:
            X_train: Training features
            y_train: Training target values
        """
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance (coefficients)
        self.feature_importance = dict(zip(X_train.columns, self.model.coef_))
        
        # Mark as trained
        self.is_trained = True
        
        # Print model information
        print(f"Model trained with {X_train.shape[0]} samples and {X_train.shape[1]} features")
        print(f"Intercept: {self.model.intercept_:.4f}")
        print("\nFeature Importance (Coefficients):")
        for feature, importance in sorted(self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature}: {importance:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            X: Features to evaluate on
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Return metrics
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance (coefficients).
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")
        
        return self.feature_importance

if __name__ == "__main__":
    # Test the model
    from src.data_loader import load_and_preprocess_data
    
    # Load data
    data = load_and_preprocess_data()
    
    # Create and train model
    model = HousePriceModel()
    model.train(data['X_train'], data['y_train'])
    
    # Evaluate model
    metrics = model.evaluate(data['X_test'], data['y_test'])
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Make sample predictions
    sample_predictions = model.predict(data['X_test'].head())
    actual_values = data['y_test'].head()
    
    # Print sample predictions
    print("\nSample Predictions:")
    for i, (pred, actual) in enumerate(zip(sample_predictions, actual_values)):
        print(f"  Sample {i+1}: Predicted = {pred:.4f}, Actual = {actual:.4f}, Diff = {abs(pred-actual):.4f}")