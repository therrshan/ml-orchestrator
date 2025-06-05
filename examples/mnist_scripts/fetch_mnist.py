#!/usr/bin/env python3
"""
Fetch MNIST data using sklearn.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    from sklearn.datasets import fetch_openml
    print("✅ sklearn imported successfully")
except ImportError:
    print("❌ sklearn not found. Install with: pip install scikit-learn")
    sys.exit(1)


def fetch_mnist_data(output_dir):
    """
    Fetch MNIST data using sklearn and save as numpy arrays.
    """
    print("📥 Fetching MNIST data from sklearn...")
    
    try:
        # Fetch MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        
        print(f"📊 MNIST data loaded:")
        print(f"   Images shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        print(f"   Classes: {np.unique(y)}")
        
        # Normalize pixel values to [0, 1]
        X = X / 255.0
        
        # Split into train/test (last 10k for test as per MNIST standard)
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        
        print(f"📈 Data split:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")
        
        # Save raw data
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'X_train.npy', X_train)
        np.save(output_dir / 'y_train.npy', y_train)
        np.save(output_dir / 'X_test.npy', X_test)
        np.save(output_dir / 'y_test.npy', y_test)
        
        # Save metadata
        metadata = {
            'dataset': 'MNIST',
            'source': 'sklearn.datasets.fetch_openml',
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X.shape[1],
            'classes': len(np.unique(y)),
            'class_names': [str(i) for i in range(10)],
            'normalized': True
        }
        
        import json
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Data saved to {output_dir}")
        print(f"   X_train.npy: {X_train.shape}")
        print(f"   y_train.npy: {y_train.shape}")
        print(f"   X_test.npy: {X_test.shape}")
        print(f"   y_test.npy: {y_test.shape}")
        print(f"   metadata.json: Dataset information")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fetching MNIST data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fetch MNIST data')
    parser.add_argument('--output', required=True, help='Output directory for raw data')
    
    args = parser.parse_args()
    
    print("📥 Starting MNIST data fetch...")
    
    success = fetch_mnist_data(args.output)
    
    if success:
        print("🎉 MNIST data fetch completed successfully!")
        return 0
    else:
        print("❌ MNIST data fetch failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())