#!/usr/bin/env python3
"""
Example script: Fetch data for ML pipeline.
This simulates downloading data from an external source.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime


def simulate_data_fetch(date_str, data_source, output_dir):
    """
    Simulate fetching data from an external source.
    
    Args:
        date_str: Date string for data to fetch
        data_source: Source identifier (e.g., 'production', 'staging')
        output_dir: Directory to save the fetched data
    """
    print(f"🔄 Fetching data for date: {date_str}")
    print(f"📊 Data source: {data_source}")
    print(f"📁 Output directory: {output_dir}")
    
    # Simulate network delay
    print("⏳ Connecting to data source...")
    time.sleep(2)
    
    # Simulate potential network issues (10% chance of retry)
    if random.random() < 0.1:
        print("❌ Network timeout, retrying...")
        time.sleep(1)
    
    # Generate sample data
    sample_data = {
        'metadata': {
            'date': date_str,
            'source': data_source,
            'fetched_at': datetime.now().isoformat(),
            'records_count': random.randint(1000, 5000)
        },
        'features': [
            {
                'user_id': i,
                'feature_1': random.uniform(0, 100),
                'feature_2': random.uniform(-50, 50),
                'feature_3': random.choice(['A', 'B', 'C']),
                'target': random.choice([0, 1])
            }
            for i in range(random.randint(100, 500))
        ]
    }
    
    # Save raw data
    raw_data_file = output_dir / 'raw_data.json'
    with open(raw_data_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(sample_data['metadata'], f, indent=2)
    
    print(f"✅ Successfully fetched {len(sample_data['features'])} records")
    print(f"💾 Data saved to: {raw_data_file}")
    print(f"📋 Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='Fetch data for ML pipeline')
    parser.add_argument('--output', required=True, help='Output directory for fetched data')
    parser.add_argument('--date', default='2024-01-01', help='Date to fetch data for')
    
    args = parser.parse_args()
    
    # Get environment variables
    data_source = os.getenv('DATA_SOURCE', 'development')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    print(f"🚀 Starting data fetch script")
    print(f"📅 Date: {args.date}")
    print(f"🎯 Log level: {log_level}")
    
    try:
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate data fetching
        simulate_data_fetch(args.date, data_source, output_dir)
        
        print("🎉 Data fetch completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during data fetch: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())