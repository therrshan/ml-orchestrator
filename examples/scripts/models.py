"""
Shared ML models for the pipeline scripts.
"""

import math
import random


class SimpleLogisticRegression:
    """
    Simple logistic regression implementation for demonstration.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.feature_names = None
        self.training_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function."""
        # Clamp z to prevent overflow/underflow
        z = max(-500, min(500, z))
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            return 1.0 if z > 0 else 0.0
    
    def fit(self, X, y, feature_names=None):
        """
        Train the logistic regression model.
        
        Args:
            X: List of feature vectors
            y: List of target values
            feature_names: List of feature names
        """
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize weights and bias
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = 0.0
        self.feature_names = feature_names or [f'feature_{i}' for i in range(n_features)]
        
        print(f"🏋️  Training model with {n_samples} samples, {n_features} features")
        
        # Training loop
        for iteration in range(self.max_iterations):
            # Forward pass
            predictions = []
            for x in X:
                z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
                predictions.append(self.sigmoid(z))
            
            # Compute loss
            loss = 0
            for pred, target in zip(predictions, y):
                # Add small epsilon to prevent log(0)
                pred = max(1e-15, min(1-1e-15, pred))
                loss += -(target * math.log(pred) + (1 - target) * math.log(1 - pred))
            loss /= n_samples
            
            # Check for convergence or numerical issues
            if math.isnan(loss) or math.isinf(loss):
                print(f"⚠️  Numerical instability detected at iteration {iteration}, stopping early")
                break
            
            # Backward pass
            dw = [0] * n_features
            db = 0
            
            for i, (x, pred, target) in enumerate(zip(X, predictions, y)):
                error = pred - target
                for j in range(n_features):
                    dw[j] += error * x[j]
                db += error
            
            # Update weights with learning rate
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j] / n_samples
            self.bias -= self.learning_rate * db / n_samples
            
            # Log progress
            if iteration % 100 == 0 or iteration < 10:
                accuracy = sum(1 for pred, target in zip(predictions, y) 
                             if (pred > 0.5) == target) / n_samples
                print(f"Iteration {iteration}: loss={loss:.4f}, accuracy={accuracy:.4f}")
                
                self.training_history.append({
                    'iteration': iteration,
                    'loss': loss,
                    'accuracy': accuracy
                })
                
                # Early stopping if loss is very low
                if loss < 0.01:
                    print(f"✅ Converged early at iteration {iteration}")
                    break
        
        print(f"✅ Training completed after {min(iteration + 1, self.max_iterations)} iterations")
    
    def predict(self, X):
        """Make predictions on new data."""
        predictions = []
        for x in X:
            z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
            predictions.append(self.sigmoid(z))
        return predictions
    
    def predict_proba(self, X):
        """Return prediction probabilities."""
        return self.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance based on weight magnitudes."""
        if self.weights is None:
            return None
        
        importance = [(name, abs(weight)) for name, weight in zip(self.feature_names, self.weights)]
        importance.sort(key=lambda x: x[1], reverse=True)
        return importance