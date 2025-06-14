#!/usr/bin/env python3
"""
Local test script - runs the ML Pipeline Orchestrator without installation.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to Python path so we can import ml_orchestrator
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can import our orchestrator
try:
    from ml_orchestrator import Pipeline, setup_logging
    print("✅ Successfully imported ml_orchestrator from local source")
except ImportError as e:
    print(f"❌ Failed to import ml_orchestrator: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def create_test_environment():
    """Create a temporary test environment."""
    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp(prefix="ml_orchestrator_local_test_"))
    print(f"🧪 Created test directory: {test_dir}")
    
    # Create necessary subdirectories
    (test_dir / "scripts").mkdir()
    (test_dir / "data").mkdir()
    (test_dir / "models").mkdir()
    (test_dir / "results").mkdir()
    (test_dir / "reports").mkdir()
    
    return test_dir


def copy_example_scripts(test_dir):
    """Copy example scripts to test directory."""
    examples_dir = Path("examples/scripts")
    scripts_dir = test_dir / "scripts"
    
    if not examples_dir.exists():
        print("❌ Examples directory not found. Make sure you have the examples/scripts/ directory.")
        return False
    
    # Copy all scripts
    copied_files = []
    for script_file in examples_dir.glob("*.py"):
        shutil.copy2(script_file, scripts_dir)
        copied_files.append(script_file.name)
        print(f"📄 Copied {script_file.name}")
    
    # Check that we have all required files
    required_files = [
        "fetch_data.py", "validate_data.py", "preprocess.py", 
        "train_model.py", "evaluate.py", "generate_report.py", "models.py"
    ]
    
    missing_files = [f for f in required_files if f not in copied_files]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Make sure you have created the models.py file in examples/scripts/")
        return False
    
    return True


def create_simple_pipeline_config(test_dir):
    """Create a simple test pipeline configuration."""
    config_content = f"""
name: "local_test_pipeline"
description: "Local test pipeline"
version: "1.0"

tasks:
  - name: "fetch_data"
    command: "python {test_dir}/scripts/fetch_data.py --output {test_dir}/data/raw/"
    environment:
      DATE: "2024-01-01"
      DATA_SOURCE: "test"
    timeout: 60
    
  - name: "validate_data"
    command: "python {test_dir}/scripts/validate_data.py --input {test_dir}/data/raw/ --output {test_dir}/data/validated/"
    depends_on: ["fetch_data"]
    timeout: 60
    
  - name: "preprocess_features"
    command: "python {test_dir}/scripts/preprocess.py --input {test_dir}/data/validated/ --output {test_dir}/data/processed/"
    depends_on: ["validate_data"]
    timeout: 120
    
  - name: "train_model"
    command: "python {test_dir}/scripts/train_model.py --data {test_dir}/data/processed/ --model {test_dir}/models/model.pkl --epochs 2"
    depends_on: ["preprocess_features"]
    timeout: 180
    retry_count: 1
    
  - name: "evaluate_model"
    command: "python {test_dir}/scripts/evaluate.py --model {test_dir}/models/model.pkl --test {test_dir}/data/processed/processed_data.json --output {test_dir}/results/"
    depends_on: ["train_model"]
    timeout: 60
    
  - name: "generate_report"
    command: "python {test_dir}/scripts/generate_report.py --results {test_dir}/results/ --output {test_dir}/reports/report.html"
    depends_on: ["evaluate_model"]
    timeout: 60
"""
    
    config_file = test_dir / "pipeline.yml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return config_file


def test_local_pipeline():
    """Test the pipeline locally without installation."""
    print("🧪 Testing ML Pipeline Orchestrator locally...")
    
    # Setup logging
    setup_logging(level='INFO')
    
    try:
        # Create test environment
        test_dir = create_test_environment()
        
        # Copy example scripts
        if not copy_example_scripts(test_dir):
            return False
        
        # Create pipeline configuration
        config_file = create_simple_pipeline_config(test_dir)
        print(f"📋 Created pipeline config: {config_file}")
        
        # Create pipeline from config
        print("📊 Creating pipeline from configuration...")
        pipeline = Pipeline.from_config(str(config_file))
        
        print(f"✅ Pipeline created: {pipeline.name}")
        print(f"🔢 Number of tasks: {len(pipeline.tasks)}")
        
        # Show execution plan
        plan = pipeline.get_execution_plan()
        print(f"📈 Execution plan ({len(plan)} levels):")
        for i, level in enumerate(plan):
            print(f"   Level {i}: {level}")
        
        # Validate pipeline
        warnings = pipeline.validate()
        if warnings:
            print("⚠️  Validation warnings:")
            for warning in warnings:
                print(f"   • {warning}")
        else:
            print("✅ Pipeline validation passed without warnings")
        
        # Run the pipeline
        print(f"\n🚀 Running pipeline: {pipeline.name}")
        print("=" * 50)
        
        success = pipeline.run()
        
        print("=" * 50)
        
        if success:
            print("🎉 Pipeline completed successfully!")
            
            # Show final status
            status = pipeline.get_status()
            print(f"📊 Final status: {status['status']}")
            print(f"✅ Completed: {status['completed_tasks']}/{status['total_tasks']} tasks")
            print(f"⏱️  Execution time: {status['execution_time']:.2f} seconds")
            
            # Check output files
            print(f"\n📁 Checking output files:")
            expected_files = [
                ("Raw data", test_dir / "data/raw/raw_data.json"),
                ("Validated data", test_dir / "data/validated/validated_data.json"),
                ("Processed data", test_dir / "data/processed/processed_data.json"),
                ("Trained model", test_dir / "models/model.pkl"),
                ("Evaluation results", test_dir / "results/evaluation_results.json"),
                ("HTML report", test_dir / "reports/report.html")
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
                print(f"\n🎉 All output files created successfully!")
                print(f"📄 View the report: file://{test_dir}/reports/report.html")
                print(f"📁 Test files location: {test_dir}")
                
                # Ask if user wants to keep the test files
                try:
                    keep = input(f"\n💾 Keep test files? (y/n): ").lower().strip()
                    if keep == 'y':
                        print(f"📁 Test files preserved at: {test_dir}")
                        return True
                except KeyboardInterrupt:
                    pass
        else:
            print("❌ Pipeline execution failed!")
            
            # Show detailed task status
            print(f"\n📊 Task Status Summary:")
            for task_name, task in pipeline.tasks.items():
                status_emoji = {
                    'success': '✅',
                    'failed': '❌', 
                    'pending': '⏳',
                    'running': '🔄'
                }.get(task.status.value, '❓')
                
                print(f"   {status_emoji} {task_name}: {task.status.value}")
                
                if task.status.value == 'failed' and task.error_message:
                    print(f"      Error: {task.error_message}")
                
                # Show the actual command that was run
                print(f"      Command: {task.command}")
                
                # If there was a result, show some details
                if task.result:
                    print(f"      Exit code: {task.result.exit_code}")
                    if task.result.stderr:
                        print(f"      Stderr: {task.result.stderr[:200]}...")
                    if task.result.stdout:
                        print(f"      Stdout: {task.result.stdout[:200]}...")
            
            # Show failed tasks
            failed_tasks = []
            for task_name, task in pipeline.tasks.items():
                if task.status.value == 'failed':
                    failed_tasks.append(task_name)
            
            if failed_tasks:
                print(f"\n💥 Failed tasks: {failed_tasks}")
            
            return False
        
        # Cleanup (unless user chose to keep files)
        print(f"\n🧹 Cleaning up test directory...")
        shutil.rmtree(test_dir)
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual components separately."""
    print("\n🔧 Testing individual components...")
    
    try:
        # Test imports
        from ml_orchestrator.core.task import Task, TaskStatus
        from ml_orchestrator.core.dag import DAG
        from ml_orchestrator.core.state import StateManager
        from ml_orchestrator.config.parser import ConfigParser
        
        print("✅ All core components imported successfully")
        
        # Test simple task creation
        task = Task(name="test_task", command="echo 'Hello World'")
        print(f"✅ Created task: {task.name}")
        
        # Test DAG creation
        tasks = [
            Task(name="task1", command="echo 'Task 1'"),
            Task(name="task2", command="echo 'Task 2'", depends_on=["task1"])
        ]
        dag = DAG(tasks)
        print(f"✅ Created DAG with {len(dag.tasks)} tasks")
        
        # Test state manager
        temp_dir = Path(tempfile.mkdtemp())
        state_manager = StateManager(str(temp_dir))
        print("✅ Created state manager")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


def main():
    """Main function."""
    print("🚀 ML Pipeline Orchestrator - Local Test")
    print("=" * 50)
    
    # Test components first
    if not test_individual_components():
        print("❌ Component tests failed")
        return False
    
    # Test full pipeline
    success = test_local_pipeline()
    
    if success:
        print("\n🎉 All tests passed! Your ML Pipeline Orchestrator is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)