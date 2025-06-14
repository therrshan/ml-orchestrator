#!/usr/bin/env python3
"""
Test MNIST pipeline locally without installation.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import orchestrator
try:
    from ml_orchestrator import Pipeline, setup_logging
    print("✅ Successfully imported ml_orchestrator from local source")
except ImportError as e:
    print(f"❌ Failed to import ml_orchestrator: {e}")
    sys.exit(1)

# Check dependencies
def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import sklearn
        print("✅ scikit-learn available")
    except ImportError:
        missing.append("scikit-learn")
    
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        missing.append("torch")
    
    try:
        import numpy
        print("✅ NumPy available")
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print(f"❌ Missing dependencies: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def create_test_environment():
    """Create test environment for MNIST pipeline."""
    test_dir = Path(tempfile.mkdtemp(prefix="mnist_test_"))
    print(f"🧪 Created test directory: {test_dir}")
    
    # Create directories
    (test_dir / "scripts").mkdir()
    (test_dir / "data" / "raw").mkdir(parents=True)
    (test_dir / "data" / "processed").mkdir(parents=True)
    (test_dir / "models").mkdir()
    (test_dir / "results").mkdir()
    (test_dir / "reports").mkdir()
    
    return test_dir


def copy_mnist_scripts(test_dir):
    """Copy MNIST scripts to test directory."""
    mnist_scripts_dir = Path("examples/mnist_scripts")
    scripts_dir = test_dir / "scripts"
    
    if not mnist_scripts_dir.exists():
        print("❌ examples/mnist_scripts directory not found")
        print("Make sure you have created all the MNIST scripts in examples/mnist_scripts/")
        return False
    
    required_scripts = [
        "fetch_mnist.py",
        "preprocess_mnist.py", 
        "train_pytorch_mnist.py",
        "evaluate_pytorch_mnist.py",
        "create_mnist_report.py"
    ]
    
    copied = []
    for script in required_scripts:
        src = mnist_scripts_dir / script
        if src.exists():
            shutil.copy2(src, scripts_dir)
            copied.append(script)
            print(f"📄 Copied {script}")
        else:
            print(f"❌ Missing script: {script}")
            return False
    
    print(f"✅ Copied {len(copied)} MNIST scripts")
    return True


def test_mnist_pipeline():
    """Test the MNIST pipeline end-to-end."""
    print("🎯 Testing MNIST Classification Pipeline")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        return False
    
    # Setup logging
    setup_logging(level='INFO')
    
    try:
        # Create test environment
        test_dir = create_test_environment()
        
        # Copy scripts
        if not copy_mnist_scripts(test_dir):
            return False
        
        # Use the existing config file from examples, but update paths
        original_config = Path("examples/mnist_pipeline.yml")
        if not original_config.exists():
            print("❌ examples/mnist_pipeline.yml not found")
            print("Make sure you have the MNIST pipeline config in examples/")
            return False
        
        # Read and modify the config to use test directory paths
        with open(original_config, 'r') as f:
            config_content = f.read()
        
        # Replace script paths in the config to point to test directory
        config_content = config_content.replace("scripts/", f"{test_dir}/scripts/")
        config_content = config_content.replace("data/", f"{test_dir}/data/")
        config_content = config_content.replace("models/", f"{test_dir}/models/")
        config_content = config_content.replace("results/", f"{test_dir}/results/")
        config_content = config_content.replace("reports/", f"{test_dir}/reports/")
        
        # Reduce epochs for faster testing
        config_content = config_content.replace("${EPOCHS:-10}", "3")
        
        # Save modified config to test directory
        config_file = test_dir / "mnist_pipeline.yml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"📋 Using pipeline config: {original_config} (modified for test)")
        
        # Create and validate pipeline
        print("🔍 Creating and validating pipeline...")
        pipeline = Pipeline.from_config(str(config_file))
        
        print(f"✅ Pipeline created: {pipeline.name}")
        print(f"🔢 Tasks: {len(pipeline.tasks)}")
        
        # Show execution plan
        plan = pipeline.get_execution_plan()
        print(f"📈 Execution plan:")
        for i, level in enumerate(plan):
            print(f"   Level {i}: {level}")
        
        # Validate pipeline
        warnings = pipeline.validate()
        if warnings:
            print("⚠️  Validation warnings:")
            for warning in warnings:
                print(f"   • {warning}")
        
        # Run pipeline
        print(f"\n🚀 Running MNIST pipeline...")
        print("🕐 This may take several minutes (downloading data + training CNN)")
        print("=" * 60)
        
        success = pipeline.run()
        
        print("=" * 60)
        
        if success:
            print("🎉 MNIST pipeline completed successfully!")
            
            # Show status
            status = pipeline.get_status()
            print(f"📊 Final status: {status['status']}")
            print(f"✅ Completed: {status['completed_tasks']}/{status['total_tasks']} tasks")
            print(f"⏱️  Execution time: {status['execution_time']:.1f} seconds")
            
            # Check output files
            print(f"\n📁 Checking output files:")
            expected_files = [
                ("Raw MNIST data", test_dir / "data/raw/X_train.npy"),
                ("Preprocessed data", test_dir / "data/processed/train.npz"),
                ("Test data", test_dir / "data/processed/test.npz"),
                ("Trained model", test_dir / "models/mnist_pytorch.pth"),
                ("Evaluation results", test_dir / "results/evaluation_results.json"),
                ("HTML report", test_dir / "reports/mnist_report.html")
            ]
            
            all_files_exist = True
            for name, file_path in expected_files:
                if file_path.exists():
                    size = file_path.stat().st_size
                    print(f"   ✅ {name}: {file_path.name} ({size:,} bytes)")
                else:
                    print(f"   ❌ {name}: {file_path.name} (missing)")
                    all_files_exist = False
            
            if all_files_exist:
                print(f"\n🎉 All MNIST pipeline outputs created successfully!")
                
                # Try to show some results
                try:
                    import json
                    results_file = test_dir / "results/evaluation_results.json"
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    print(f"\n📊 MNIST Results Summary:")
                    print(f"   🎯 Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
                    print(f"   📈 Test Samples: {results['test_samples']:,}")
                    print(f"   🔢 Model Parameters: {results['model_info']['num_parameters']:,}")
                    print(f"   💪 Training Epochs: {results['model_info']['epochs_trained']}")
                    print(f"   ⚡ Device Used: {results['evaluation_metadata']['device']}")
                    
                    # Show per-digit performance
                    per_class_acc = results['per_class_accuracy']
                    best_digit = per_class_acc.index(max(per_class_acc))
                    worst_digit = per_class_acc.index(min(per_class_acc))
                    
                    print(f"   🏆 Best digit: {best_digit} ({per_class_acc[best_digit]:.3f})")
                    print(f"   🔍 Hardest digit: {worst_digit} ({per_class_acc[worst_digit]:.3f})")
                    
                except Exception as e:
                    print(f"   ⚠️  Could not load detailed results: {e}")
                
                print(f"\n📄 View the beautiful report: file://{test_dir}/reports/mnist_report.html")
                print(f"📁 All files location: {test_dir}")
                
                # Ask if user wants to keep files
                try:
                    keep = input(f"\n💾 Keep MNIST test files for inspection? (y/n): ").lower().strip()
                    if keep == 'y':
                        print(f"📁 MNIST test files preserved at: {test_dir}")
                        return True
                except KeyboardInterrupt:
                    pass
            else:
                print(f"\n⚠️  Some files are missing, but pipeline may have partially completed")
        
        else:
            print("❌ MNIST pipeline execution failed!")
            
            # Show detailed task status
            print(f"\n📊 Task Status:")
            for task_name, task in pipeline.tasks.items():
                status_emoji = {
                    'success': '✅',
                    'failed': '❌', 
                    'pending': '⏳',
                    'running': '🔄'
                }.get(task.status.value, '❓')
                
                print(f"   {status_emoji} {task_name}: {task.status.value}")
                
                if task.status.value == 'failed':
                    if task.error_message:
                        print(f"      💥 Error: {task.error_message}")
                    if task.result and task.result.stderr:
                        print(f"      📝 Stderr: {task.result.stderr[:200]}...")
        
        # Cleanup unless user chose to keep
        print(f"\n🧹 Cleaning up test directory...")
        shutil.rmtree(test_dir)
        
        return success
        
    except Exception as e:
        print(f"❌ MNIST test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_requirements():
    """Show requirements for MNIST pipeline."""
    print("📋 MNIST Pipeline Requirements:")
    print("   • Python 3.8+")
    print("   • scikit-learn (for MNIST data)")
    print("   • PyTorch (for CNN model)")
    print("   • NumPy (for data processing)")
    print()
    print("📦 Install dependencies:")
    print("   pip install scikit-learn torch numpy")
    print()


def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--requirements":
        show_requirements()
        return True
    
    print("🎯 ML Pipeline Orchestrator - MNIST Test")
    print("=" * 50)
    
    success = test_mnist_pipeline()
    
    if success:
        print("\n🎉 MNIST pipeline test completed successfully!")
     
    else:
        print("\n❌ MNIST pipeline test failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)