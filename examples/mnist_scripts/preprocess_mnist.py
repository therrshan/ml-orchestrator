#!/usr/bin/env python3
"""
Preprocess MNIST data for PyTorch training.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path


def preprocess_mnist_data(input_dir, output_dir):
    """
    Preprocess MNIST data for PyTorch.
    """
    print("⚙️ Preprocessing MNIST data...")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load raw data
        print("📂 Loading raw data...")
        X_train = np.load(input_dir / 'X_train.npy')
        y_train = np.load(input_dir / 'y_train.npy')
        X_test = np.load(input_dir / 'X_test.npy')
        y_test = np.load(input_dir / 'y_test.npy')
        
        print(f"📊 Loaded data shapes:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   X_test: {X_test.shape}")
        print(f"   y_test: {y_test.shape}")
        
        # Reshape for CNN: (N, 28, 28) -> (N, 1, 28, 28)
        X_train_reshaped = X_train.reshape(-1, 28, 28)
        X_test_reshaped = X_test.reshape(-1, 28, 28)
        
        print(f"🔄 Reshaped for CNN:")
        print(f"   X_train: {X_train_reshaped.shape}")
        print(f"   X_test: {X_test_reshaped.shape}")
        
        # Create validation split (10% of training data)
        val_size = int(0.1 * len(X_train_reshaped))
        indices = np.random.permutation(len(X_train_reshaped))
        
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        
        X_train_final = X_train_reshaped[train_idx]
        y_train_final = y_train[train_idx]
        X_val = X_train_reshaped[val_idx]
        y_val = y_train[val_idx]
        
        print(f"📈 Train/Validation split:")
        print(f"   Training: {len(X_train_final)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test_reshaped)} samples")
        
        # Save processed data
        np.savez_compressed(
            output_dir / 'train.npz',
            X=X_train_final,
            y=y_train_final
        )
        
        np.savez_compressed(
            output_dir / 'validation.npz',
            X=X_val,
            y=y_val
        )
        
        np.savez_compressed(
            output_dir / 'test.npz',
            X=X_test_reshaped,
            y=y_test
        )
        
        # Save preprocessing info
        preprocessing_info = {
            'input_shape': [1, 28, 28],  # Channels, Height, Width
            'num_classes': 10,
            'train_samples': len(X_train_final),
            'val_samples': len(X_val),
            'test_samples': len(X_test_reshaped),
            'pixel_range': [0.0, 1.0],
            'mean_pixel': float(X_train_final.mean()),
            'std_pixel': float(X_train_final.std()),
            'class_distribution': {
                str(i): int(np.sum(y_train_final == i)) 
                for i in range(10)
            }
        }
        
        import json
        with open(output_dir / 'preprocessing_info.json', 'w') as f:
            json.dump(preprocessing_info, f, indent=2)
        
        print(f"💾 Processed data saved:")
        print(f"   train.npz: {len(X_train_final)} samples")
        print(f"   validation.npz: {len(X_val)} samples")
        print(f"   test.npz: {len(X_test_reshaped)} samples")
        print(f"   preprocessing_info.json: Metadata")
        
        # Show class distribution
        print(f"📊 Class distribution in training set:")
        for digit, count in preprocessing_info['class_distribution'].items():
            print(f"   Digit {digit}: {count} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error preprocessing data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Preprocess MNIST data')
    parser.add_argument('--input', required=True, help='Input directory with raw data')
    parser.add_argument('--output', required=True, help='Output directory for processed data')
    
    args = parser.parse_args()
    
    print("⚙️ Starting MNIST preprocessing...")
    
    # Set random seed for reproducible splits
    np.random.seed(42)
    
    success = preprocess_mnist_data(args.input, args.output)
    
    if success:
        print("🎉 MNIST preprocessing completed successfully!")
        return 0
    else:
        print("❌ MNIST preprocessing failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())