#!/usr/bin/env python3
"""
Evaluate PyTorch MNIST model.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import classification_report, confusion_matrix
    print("✅ PyTorch and sklearn imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install with: pip install torch scikit-learn")
    sys.exit(1)


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification - must match training script."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Conv block 3
        x = F.relu(self.conv3(x))  # 7x7 -> 7x7
        x = self.pool(x)  # 7x7 -> 3x3
        
        # Flatten and fully connected
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_model(model_path):
    """Load the trained PyTorch model."""
    print(f"📂 Loading model from {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model and load state dict
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully")
    return model, checkpoint


def load_test_data(test_data_path):
    """Load test data."""
    print(f"📂 Loading test data from {test_data_path}")
    
    test_data = np.load(test_data_path)
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"📊 Test data shape: {X_test.shape}")
    print(f"📊 Test labels shape: {y_test.shape}")
    
    return X_test, y_test


def evaluate_model(model, X_test, y_test, batch_size=64):
    """Evaluate model on test data."""
    print("🧪 Evaluating model on test data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create test dataset and loader
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.LongTensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluation
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = (all_predictions == all_targets).mean()
    
    print(f"📈 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return all_predictions, all_probabilities, all_targets, accuracy


def generate_detailed_metrics(y_true, y_pred, y_proba):
    """Generate detailed evaluation metrics."""
    
    # Classification report
    class_names = [str(i) for i in range(10)]
    class_report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    # Confidence statistics
    max_probs = np.max(y_proba, axis=1)
    avg_confidence = np.mean(max_probs)
    
    return {
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'per_class_accuracy': per_class_acc.tolist(),
        'average_confidence': float(avg_confidence),
        'confidence_stats': {
            'mean': float(np.mean(max_probs)),
            'std': float(np.std(max_probs)),
            'min': float(np.min(max_probs)),
            'max': float(np.max(max_probs))
        }
    }


def save_results(results, predictions, probabilities, targets, output_dir):
    """Save evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions and probabilities
    np.savez_compressed(
        output_dir / 'predictions.npz',
        predictions=predictions,
        probabilities=probabilities,
        targets=targets
    )
    
    print(f"💾 Results saved to {output_dir}")
    print(f"   evaluation_results.json: Detailed metrics")
    print(f"   predictions.npz: Predictions and probabilities")


def print_summary(results):
    """Print evaluation summary."""
    print("\n" + "="*50)
    print("📊 EVALUATION SUMMARY")
    print("="*50)
    
    overall = results['classification_report']['macro avg']
    print(f"Overall Accuracy: {results['classification_report']['accuracy']:.4f}")
    print(f"Macro Avg Precision: {overall['precision']:.4f}")
    print(f"Macro Avg Recall: {overall['recall']:.4f}")
    print(f"Macro Avg F1-Score: {overall['f1-score']:.4f}")
    print(f"Average Confidence: {results['average_confidence']:.4f}")
    
    print(f"\n📈 Per-Class Accuracy:")
    for i, acc in enumerate(results['per_class_accuracy']):
        print(f"   Digit {i}: {acc:.4f} ({acc*100:.1f}%)")
    
    # Find best and worst performing classes
    per_class_acc = np.array(results['per_class_accuracy'])
    best_class = np.argmax(per_class_acc)
    worst_class = np.argmin(per_class_acc)
    
    print(f"\n🏆 Best performing digit: {best_class} ({per_class_acc[best_class]:.4f})")
    print(f"🔍 Worst performing digit: {worst_class} ({per_class_acc[worst_class]:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PyTorch MNIST model')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--test', required=True, help='Path to test data')
    parser.add_argument('--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🧪 Starting PyTorch MNIST evaluation...")
    
    try:
        # Load model
        model, checkpoint = load_model(args.model)
        
        # Load test data
        X_test, y_test = load_test_data(args.test)
        
        # Evaluate model
        predictions, probabilities, targets, accuracy = evaluate_model(model, X_test, y_test)
        
        # Generate detailed metrics
        detailed_metrics = generate_detailed_metrics(targets, predictions, probabilities)
        
        # Combine results
        results = {
            'model_info': checkpoint.get('model_info', {}),
            'test_accuracy': float(accuracy),
            'test_samples': len(targets),
            **detailed_metrics,
            'evaluation_metadata': {
                'model_path': str(args.model),
                'test_data_path': str(args.test),
                'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            }
        }
        
        # Save results
        save_results(results, predictions, probabilities, targets, args.output)
        
        # Print summary
        print_summary(results)
        
        print("\n🎉 Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())