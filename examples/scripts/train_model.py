#!/usr/bin/env python3
"""
Example script: Train ML model.
This trains a simple model on the processed data.
"""

import os
import sys
import json
import pickle
import argparse
import random
from pathlib import Path

# Import the shared model class
from models import SimpleLogisticRegression


def extract_features(data, feature_names):
    """
    Extract numerical features from data records.
    
    Args:
        data: List of data records
        feature_names: List of feature names to extract
        
    Returns:
        Tuple of (feature_matrix, targets)
    """
    X = []
    y = []
    
    for record in data:
        # Extract features
        features = []
        for feature_name in feature_names:
            if feature_name in record:
                features.append(record[feature_name])
            else:
                features.append(0.0)  # Default value for missing features
        
        X.append(features)
        y.append(record.get('target', 0))
    
    return X, y


def train_model(data_dir, model_path, epochs):
    """
    Train the ML model.
    
    Args:
        data_dir: Directory containing processed data
        model_path: Path to save the trained model
        epochs: Number of training epochs
    """
    print(f"🤖 Training model from data in: {data_dir}")
    
    # Load processed data
    processed_file = data_dir / 'processed_data.json'
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed data file not found: {processed_file}")
    
    with open(processed_file, 'r') as f:
        data = json.load(f)
    
    train_data = data['train_features']
    print(f"📊 Training samples: {len(train_data)}")
    
    # Define features to use for training
    numerical_features = [
        'feature_1_norm', 'feature_2_norm', 'feature_1_squared_norm',
        'feature_2_abs_norm', 'feature_interaction_norm', 'feature_ratio_norm',
        'feature_3_A', 'feature_3_B', 'feature_3_C'
    ]
    
    # Extract features and targets
    X_train, y_train = extract_features(train_data, numerical_features)
    
    print(f"🎯 Features used: {len(numerical_features)}")
    print(f"🏷️  Target distribution: {sum(y_train)}/{len(y_train)} positive")
    
    # Create and train model
    model = SimpleLogisticRegression(
        learning_rate=0.01,
        max_iterations=min(epochs * 10, 1000)  # Scale epochs
    )
    
    model.fit(X_train, y_train, numerical_features)
    
    # Feature importance
    importance = model.get_feature_importance()
    print("\n📊 Feature Importance:")
    for name, weight in importance[:5]:  # Top 5 features
        print(f"   {name}: {weight:.4f}")
    
    # Model evaluation on training data
    train_predictions = model.predict(X_train)
    train_accuracy = sum(1 for pred, target in zip(train_predictions, y_train) 
                        if (pred > 0.5) == target) / len(y_train)
    
    print(f"\n📈 Training Accuracy: {train_accuracy:.4f}")
    
    # Save model
    model_data = {
        'model': model,
        'feature_names': numerical_features,
        'training_accuracy': train_accuracy,
        'feature_importance': importance,
        'training_history': model.training_history,
        'metadata': {
            'trained_at': __import__('datetime').datetime.now().isoformat(),
            'training_samples': len(train_data),
            'epochs': epochs,
            'model_type': 'SimpleLogisticRegression'
        }
    }
    
    # Create model directory
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Save model metadata as JSON for easy inspection
    metadata_file = model_path.with_suffix('.json')
    metadata = model_data.copy()
    del metadata['model']  # Remove unpicklable object
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Model saved to: {model_path}")
    print(f"📋 Model metadata saved to: {metadata_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--data', required=True, help='Directory with processed data')
    parser.add_argument('--model', required=True, help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Get environment variables
    cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES', 'not set')
    
    print(f"🤖 Starting model training script")
    print(f"📊 Data directory: {args.data}")
    print(f"💾 Model path: {args.model}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"🖥️  CUDA devices: {cuda_devices}")
    
    try:
        # Set random seed for reproducibility
        random.seed(42)
        
        # Train model
        success = train_model(Path(args.data), args.model, args.epochs)
        
        if success:
            print("🎉 Model training completed successfully!")
            return 0
        else:
            print("❌ Model training failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())