#!/usr/bin/env python3
"""
Debug test - tests individual components step by step.
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_orchestrator import Pipeline, setup_logging


def test_individual_script(script_path, args, working_dir=None):
    """Test running an individual script."""
    print(f"🧪 Testing script: {script_path}")
    print(f"   Args: {args}")
    
    try:
        cmd = [sys.executable, str(script_path)] + args
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"   Exit code: {result.returncode}")
        if result.stdout:
            print(f"   Stdout: {result.stdout[:300]}...")
        if result.stderr:
            print(f"   Stderr: {result.stderr[:300]}...")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("   ❌ Script timed out")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def debug_pipeline_step_by_step():
    """Debug the pipeline by testing each step individually."""
    print("🔍 Debug Test - Step by Step Pipeline")
    print("=" * 50)
    
    # Setup
    test_dir = Path(tempfile.mkdtemp(prefix="debug_test_"))
    print(f"📁 Test directory: {test_dir}")
    
    # Create directories
    (test_dir / "scripts").mkdir()
    (test_dir / "data" / "raw").mkdir(parents=True)
    (test_dir / "data" / "validated").mkdir(parents=True) 
    (test_dir / "data" / "processed").mkdir(parents=True)
    (test_dir / "models").mkdir()
    (test_dir / "results").mkdir()
    (test_dir / "reports").mkdir()
    
    # Copy scripts
    examples_dir = Path("examples/scripts")
    if not examples_dir.exists():
        print("❌ examples/scripts directory not found")
        return False
    
    for script in examples_dir.glob("*.py"):
        shutil.copy2(script, test_dir / "scripts")
        print(f"📄 Copied {script.name}")
    
    # Make sure we have all required scripts
    required_scripts = [
        "fetch_data.py", "validate_data.py", "preprocess.py", 
        "train_model.py", "evaluate.py", "generate_report.py", "models.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not (test_dir / "scripts" / script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ Missing required scripts: {missing_scripts}")
        return False
    
    # Test each step individually
    scripts_dir = test_dir / "scripts"
    
    # Step 1: Fetch data
    print(f"\n{'='*20} Step 1: Fetch Data {'='*20}")
    success = test_individual_script(
        scripts_dir / "fetch_data.py",
        ["--output", str(test_dir / "data/raw/")]
    )
    if not success:
        print("❌ Fetch data failed")
        return False
    
    # Check output
    raw_file = test_dir / "data/raw/raw_data.json"
    if raw_file.exists():
        print(f"✅ Raw data created: {raw_file.stat().st_size} bytes")
    else:
        print("❌ Raw data file not created")
        return False
    
    # Step 2: Validate data
    print(f"\n{'='*20} Step 2: Validate Data {'='*20}")
    success = test_individual_script(
        scripts_dir / "validate_data.py",
        ["--input", str(test_dir / "data/raw/"), "--output", str(test_dir / "data/validated/")]
    )
    if not success:
        print("❌ Validate data failed")
        return False
    
    # Check output
    validated_file = test_dir / "data/validated/validated_data.json"
    if validated_file.exists():
        print(f"✅ Validated data created: {validated_file.stat().st_size} bytes")
    else:
        print("❌ Validated data file not created")
        return False
    
    # Step 3: Preprocess
    print(f"\n{'='*20} Step 3: Preprocess {'='*20}")
    success = test_individual_script(
        scripts_dir / "preprocess.py",
        ["--input", str(test_dir / "data/validated/"), "--output", str(test_dir / "data/processed/")]
    )
    if not success:
        print("❌ Preprocess failed")
        return False
    
    # Check output
    processed_file = test_dir / "data/processed/processed_data.json"
    if processed_file.exists():
        print(f"✅ Processed data created: {processed_file.stat().st_size} bytes")
    else:
        print("❌ Processed data file not created")
        return False
    
    # Step 4: Train model (with minimal epochs)
    print(f"\n{'='*20} Step 4: Train Model {'='*20}")
    success = test_individual_script(
        scripts_dir / "train_model.py",
        ["--data", str(test_dir / "data/processed/"), 
         "--model", str(test_dir / "models/model.pkl"), 
         "--epochs", "2"]
    )
    if not success:
        print("❌ Train model failed")
        return False
    
    # Check output
    model_file = test_dir / "models/model.pkl"
    if model_file.exists():
        print(f"✅ Model created: {model_file.stat().st_size} bytes")
    else:
        print("❌ Model file not created")
        return False
    
    # Step 5: Evaluate model
    print(f"\n{'='*20} Step 5: Evaluate Model {'='*20}")
    success = test_individual_script(
        scripts_dir / "evaluate.py",
        ["--model", str(test_dir / "models/model.pkl"),
         "--test", str(test_dir / "data/processed/processed_data.json"),
         "--output", str(test_dir / "results/")]
    )
    if not success:
        print("❌ Evaluate model failed")
        return False
    
    # Check output
    results_file = test_dir / "results/evaluation_results.json"
    if results_file.exists():
        print(f"✅ Evaluation results created: {results_file.stat().st_size} bytes")
    else:
        print("❌ Evaluation results file not created")
        return False
    
    # Step 6: Generate report
    print(f"\n{'='*20} Step 6: Generate Report {'='*20}")
    success = test_individual_script(
        scripts_dir / "generate_report.py",
        ["--results", str(test_dir / "results/"),
         "--output", str(test_dir / "reports/report.html")]
    )
    if not success:
        print("❌ Generate report failed")
        return False
    
    # Check output
    report_file = test_dir / "reports/report.html"
    if report_file.exists():
        print(f"✅ Report created: {report_file.stat().st_size} bytes")
    else:
        print("❌ Report file not created")
        return False
    
    print(f"\n🎉 All individual steps completed successfully!")
    print(f"📁 Files are in: {test_dir}")
    
    # Ask if user wants to keep files for inspection
    try:
        keep = input("\n💾 Keep test files for inspection? (y/n): ").lower().strip()
        if keep == 'y':
            print(f"📁 Test files kept at: {test_dir}")
            return True
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    print("🧹 Cleaning up...")
    shutil.rmtree(test_dir)
    return True


if __name__ == "__main__":
    setup_logging(level='INFO')
    success = debug_pipeline_step_by_step()
    sys.exit(0 if success else 1)