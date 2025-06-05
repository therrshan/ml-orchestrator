#!/usr/bin/env python3
"""
Example script: Evaluate trained model.
This evaluates the model performance on test data.
"""

import os
import sys
import json
import pickle
import argparse
import math
from pathlib import Path

# Import the shared model class
from models import SimpleLogisticRegression


def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    n = len(y_true)
    
    # Basic metrics
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    
    # Derived metrics
    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Log loss
    log_loss = 0
    for true, proba in zip(y_true, y_proba):
        log_loss += -(true * math.log(proba + 1e-15) + (1 - true) * math.log(1 - proba + 1e-15))
    log_loss /= n
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'log_loss': log_loss,
        'confusion_matrix': {
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn
        },
        'support': {
            'positive': sum(y_true),
            'negative': n - sum(y_true),
            'total': n
        }
    }


def load_test_data(data_path):
    """
    Load test data from file.
    
    Args:
        data_path: Path to test data file or directory
        
    Returns:
        List of test records
    """
    data_path = Path(data_path)
    
    # Try different possible file locations
    possible_files = []
    
    if data_path.is_dir():
        # If it's a directory, look for test files
        possible_files = [
            data_path / 'test.json',
            data_path / 'processed_data.json'
        ]
    else:
        # If it's a file path, use it directly
        possible_files = [data_path]
    
    for file_path in possible_files:
        if file_path.exists():
            print(f"📁 Loading test data from: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract test features based on data structure
                if 'test_features' in data:
                    print(f"📊 Found test_features with {len(data['test_features'])} samples")
                    return data['test_features']
                elif 'features' in data:
                    print(f"📊 Found features with {len(data['features'])} samples")
                    return data['features']
                elif isinstance(data, list):
                    print(f"📊 Found list data with {len(data)} samples")
                    return data
                else:
                    print(f"📊 Using data as-is with {len(data)} keys")
                    return data
                    
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON from {file_path}: {e}")
                continue
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
    
    raise FileNotFoundError(f"No valid test data found. Tried: {[str(f) for f in possible_files]}")


def evaluate_model(model_path, test_data_path, output_dir):
    """
    Evaluate the trained model.
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to test data
        output_dir: Directory to save evaluation results
    """
    print(f"📊 Evaluating model: {model_path}")
    print(f"🧪 Test data: {test_data_path}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    print(f"🤖 Model type: {model_data['metadata']['model_type']}")
    print(f"🏋️  Training accuracy: {model_data['training_accuracy']:.4f}")
    
    # Load test data
    test_data = load_test_data(test_data_path)
    print(f"🧪 Test samples: {len(test_data)}")
    
    # Extract features and targets
    X_test = []
    y_test = []
    
    for record in test_data:
        features = []
        for feature_name in feature_names:
            features.append(record.get(feature_name, 0.0))
        X_test.append(features)
        y_test.append(record.get('target', 0))
    
    # Make predictions
    print("🔮 Making predictions...")
    y_proba = model.predict_proba(X_test)
    y_pred = [1 if p > 0.5 else 0 for p in y_proba]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Print results
    print("\n📈 Evaluation Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1_score']:.4f}")
    print(f"   Log Loss:  {metrics['log_loss']:.4f}")
    
    print("\n📊 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   True Positive:  {cm['true_positive']}")
    print(f"   True Negative:  {cm['true_negative']}")
    print(f"   False Positive: {cm['false_positive']}")
    print(f"   False Negative: {cm['false_negative']}")
    
    # Feature importance from model
    print("\n🎯 Top Feature Importance:")
    for name, importance in model_data['feature_importance'][:5]:
        print(f"   {name}: {importance:.4f}")
    
    # Create evaluation report
    evaluation_report = {
        'model_info': {
            'model_path': str(model_path),
            'model_type': model_data['metadata']['model_type'],
            'training_accuracy': model_data['training_accuracy'],
            'feature_count': len(feature_names)
        },
        'test_info': {
            'test_data_path': str(test_data_path),
            'test_samples': len(test_data),
            'positive_samples': metrics['support']['positive'],
            'negative_samples': metrics['support']['negative']
        },
        'metrics': metrics,
        'feature_importance': model_data['feature_importance'],
        'evaluation_metadata': {
            'evaluated_at': __import__('datetime').datetime.now().isoformat(),
            'threshold': 0.5
        }
    }
    
    # Save evaluation results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    # Save predictions
    predictions_file = output_dir / 'predictions.json'
    predictions_data = {
        'predictions': [
            {
                'true_label': true,
                'predicted_label': pred,
                'predicted_probability': prob
            }
            for true, pred, prob in zip(y_test, y_pred, y_proba)
        ]
    }
    
    with open(predictions_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    print(f"\n💾 Evaluation results saved to: {results_file}")
    print(f"🔮 Predictions saved to: {predictions_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--test', required=True, help='Path to test data')
    parser.add_argument('--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"📊 Starting model evaluation script")
    
    try:
        # Evaluate model
        success = evaluate_model(args.model, args.test, args.output)
        
        if success:
            print("🎉 Model evaluation completed successfully!")
            return 0
        else:
            print("❌ Model evaluation failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during model evaluation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())