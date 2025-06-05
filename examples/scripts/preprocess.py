#!/usr/bin/env python3
"""
Example script: Preprocess and engineer features.
This performs feature engineering and data preprocessing.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
import math


def engineer_features(data):
    """
    Perform feature engineering on the data.
    
    Args:
        data: Dictionary containing validated data
        
    Returns:
        Dictionary with engineered features
    """
    print("🔧 Engineering features...")
    
    features = data['features']
    engineered_features = []
    
    for record in features:
        # Original features
        new_record = record.copy()
        
        # Feature engineering
        f1 = record['feature_1']
        f2 = record['feature_2']
        
        # Create new features
        new_record['feature_1_squared'] = f1 * f1
        new_record['feature_2_abs'] = abs(f2)
        new_record['feature_interaction'] = f1 * abs(f2)
        new_record['feature_ratio'] = f1 / (abs(f2) + 1e-6)  # Avoid division by zero
        
        # Binning feature_1
        if f1 < 25:
            new_record['feature_1_bin'] = 'low'
        elif f1 < 75:
            new_record['feature_1_bin'] = 'medium'
        else:
            new_record['feature_1_bin'] = 'high'
        
        # One-hot encode feature_3
        new_record['feature_3_A'] = 1 if record['feature_3'] == 'A' else 0
        new_record['feature_3_B'] = 1 if record['feature_3'] == 'B' else 0
        new_record['feature_3_C'] = 1 if record['feature_3'] == 'C' else 0
        
        engineered_features.append(new_record)
    
    return engineered_features


def normalize_features(features):
    """
    Normalize numerical features.
    
    Args:
        features: List of feature dictionaries
        
    Returns:
        Tuple of (normalized_features, normalization_stats)
    """
    print("📏 Normalizing features...")
    
    # Calculate statistics for numerical features
    numerical_features = ['feature_1', 'feature_2', 'feature_1_squared', 
                         'feature_2_abs', 'feature_interaction', 'feature_ratio']
    
    stats = {}
    for feature_name in numerical_features:
        values = [record[feature_name] for record in features if feature_name in record]
        if values:
            stats[feature_name] = {
                'mean': sum(values) / len(values),
                'std': math.sqrt(sum((x - sum(values) / len(values))**2 for x in values) / len(values))
            }
    
    # Normalize features
    normalized_features = []
    for record in features:
        new_record = record.copy()
        
        for feature_name in numerical_features:
            if feature_name in record and feature_name in stats:
                mean = stats[feature_name]['mean']
                std = stats[feature_name]['std']
                if std > 0:
                    new_record[f'{feature_name}_norm'] = (record[feature_name] - mean) / std
                else:
                    new_record[f'{feature_name}_norm'] = 0
        
        normalized_features.append(new_record)
    
    return normalized_features, stats


def create_train_test_split(features, test_ratio=0.2):
    """
    Split data into training and testing sets.
    
    Args:
        features: List of feature dictionaries
        test_ratio: Ratio of data to use for testing
        
    Returns:
        Tuple of (train_features, test_features)
    """
    print(f"✂️  Splitting data (test ratio: {test_ratio})")
    
    # Shuffle data
    shuffled_features = features.copy()
    random.shuffle(shuffled_features)
    
    # Split
    split_idx = int(len(shuffled_features) * (1 - test_ratio))
    train_features = shuffled_features[:split_idx]
    test_features = shuffled_features[split_idx:]
    
    print(f"📊 Train samples: {len(train_features)}")
    print(f"📊 Test samples: {len(test_features)}")
    
    return train_features, test_features


def preprocess_data(input_dir, output_dir):
    """
    Complete preprocessing pipeline.
    
    Args:
        input_dir: Directory containing validated data
        output_dir: Directory to save processed data
    """
    print(f"⚙️  Processing data from: {input_dir}")
    
    # Load validated data
    validated_file = input_dir / 'validated_data.json'
    if not validated_file.exists():
        raise FileNotFoundError(f"Validated data file not found: {validated_file}")
    
    with open(validated_file, 'r') as f:
        validated_data = json.load(f)
    
    original_count = len(validated_data['features'])
    print(f"📊 Processing {original_count} records")
    
    # Feature engineering
    engineered_features = engineer_features(validated_data)
    
    # Normalization
    normalized_features, normalization_stats = normalize_features(engineered_features)
    
    # Train/test split
    train_features, test_features = create_train_test_split(normalized_features)
    
    # Create processed data structure
    processed_data = {
        'metadata': validated_data['metadata'],
        'preprocessing': {
            'feature_engineering': True,
            'normalization': True,
            'train_test_split': True,
            'test_ratio': 0.2,
            'normalization_stats': normalization_stats,
            'processed_at': __import__('datetime').datetime.now().isoformat()
        },
        'train_features': train_features,
        'test_features': test_features
    }
    
    # Save processed data
    processed_file = output_dir / 'processed_data.json'
    with open(processed_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    # Save separate train and test files for convenience
    train_file = output_dir / 'train.json'
    with open(train_file, 'w') as f:
        json.dump({'features': train_features}, f, indent=2)
    
    test_file = output_dir / 'test.json'
    with open(test_file, 'w') as f:
        json.dump({'features': test_features}, f, indent=2)
    
    # Save preprocessing stats
    stats_file = output_dir / 'preprocessing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(processed_data['preprocessing'], f, indent=2)
    
    print(f"✅ Preprocessing completed")
    print(f"💾 Processed data saved to: {processed_file}")
    print(f"🚂 Training data: {train_file}")
    print(f"🧪 Test data: {test_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Preprocess and engineer features')
    parser.add_argument('--input', required=True, help='Input directory with validated data')
    parser.add_argument('--output', required=True, help='Output directory for processed data')
    
    args = parser.parse_args()
    
    print(f"⚙️  Starting preprocessing script")
    
    try:
        # Create directories
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Process data
        success = preprocess_data(input_dir, output_dir)
        
        if success:
            print("🎉 Preprocessing completed successfully!")
            return 0
        else:
            print("❌ Preprocessing failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())