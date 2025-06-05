#!/usr/bin/env python3
"""
Example script: Validate fetched data.
This performs data quality checks and validation.
"""

import os
import sys
import json
import argparse
from pathlib import Path


def validate_data_quality(data):
    """
    Perform data quality checks.
    
    Args:
        data: Dictionary containing the data to validate
        
    Returns:
        Tuple of (is_valid, issues_found)
    """
    issues = []
    
    # Check if data has required structure
    if 'features' not in data:
        issues.append("Missing 'features' field in data")
        return False, issues
    
    if 'metadata' not in data:
        issues.append("Missing 'metadata' field in data")
    
    features = data['features']
    
    # Check if we have enough records
    if len(features) < 50:
        issues.append(f"Insufficient data: only {len(features)} records (minimum 50)")
    
    # Validate feature schema
    required_fields = ['user_id', 'feature_1', 'feature_2', 'feature_3', 'target']
    
    for i, record in enumerate(features[:10]):  # Check first 10 records
        for field in required_fields:
            if field not in record:
                issues.append(f"Record {i} missing required field: {field}")
        
        # Validate data types and ranges
        if 'feature_1' in record:
            if not isinstance(record['feature_1'], (int, float)) or record['feature_1'] < 0:
                issues.append(f"Record {i}: feature_1 should be positive number")
        
        if 'target' in record:
            if record['target'] not in [0, 1]:
                issues.append(f"Record {i}: target should be 0 or 1")
    
    # Check for duplicate user_ids
    user_ids = [record.get('user_id') for record in features]
    if len(set(user_ids)) != len(user_ids):
        issues.append("Duplicate user_ids found in data")
    
    # Check data distribution
    targets = [record.get('target') for record in features if 'target' in record]
    if targets:
        positive_ratio = sum(targets) / len(targets)
        if positive_ratio < 0.05 or positive_ratio > 0.95:
            issues.append(f"Imbalanced target distribution: {positive_ratio:.2%} positive")
    
    return len(issues) == 0, issues


def clean_and_validate_data(input_dir, output_dir):
    """
    Load, validate, and clean the data.
    
    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save validated data
    """
    print(f"🔍 Validating data from: {input_dir}")
    
    # Load raw data
    raw_data_file = input_dir / 'raw_data.json'
    if not raw_data_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_data_file}")
    
    with open(raw_data_file, 'r') as f:
        raw_data = json.load(f)
    
    print(f"📊 Loaded {len(raw_data.get('features', []))} records")
    
    # Validate data quality
    is_valid, issues = validate_data_quality(raw_data)
    
    if issues:
        print("⚠️  Data quality issues found:")
        for issue in issues:
            print(f"   • {issue}")
    
    if not is_valid:
        print("❌ Data validation failed!")
        return False
    
    # Clean the data (remove invalid records)
    cleaned_features = []
    for record in raw_data['features']:
        # Basic cleaning - remove records with missing critical fields
        if all(field in record for field in ['user_id', 'feature_1', 'feature_2', 'target']):
            cleaned_features.append(record)
    
    cleaned_data = {
        'metadata': raw_data['metadata'],
        'features': cleaned_features,
        'validation': {
            'original_count': len(raw_data['features']),
            'cleaned_count': len(cleaned_features),
            'issues_found': issues,
            'validated_at': __import__('datetime').datetime.now().isoformat()
        }
    }
    
    # Save validated data
    validated_file = output_dir / 'validated_data.json'
    with open(validated_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Save validation report
    report_file = output_dir / 'validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(cleaned_data['validation'], f, indent=2)
    
    print(f"✅ Data validation completed")
    print(f"📈 Records: {len(raw_data['features'])} → {len(cleaned_features)}")
    print(f"💾 Validated data saved to: {validated_file}")
    print(f"📋 Validation report saved to: {report_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate data quality')
    parser.add_argument('--input', required=True, help='Input directory with raw data')
    parser.add_argument('--output', required=True, help='Output directory for validated data')
    
    args = parser.parse_args()
    
    print(f"🔍 Starting data validation script")
    
    try:
        # Create directories
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate data
        success = clean_and_validate_data(input_dir, output_dir)
        
        if success:
            print("🎉 Data validation completed successfully!")
            return 0
        else:
            print("❌ Data validation failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during data validation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())