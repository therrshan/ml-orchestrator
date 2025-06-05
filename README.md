# ML Pipeline Orchestrator

A lightweight, production-ready ML pipeline orchestrator that transforms your individual ML scripts into reliable, automated workflows. Built specifically for machine learning teams who need robust pipeline management without the complexity of Apache Airflow.

## ✨ Key Features

- **🔄 Smart Dependency Management** - Automatic task ordering with DAG validation
- **💾 State Persistence** - Resume interrupted pipelines from where they left off
- **🔁 Robust Error Handling** - Configurable retry logic and graceful failure recovery
- **⚙️ CLI & Python API** - Use from command line or integrate into your code
- **📊 Real ML Examples** - MNIST computer vision pipeline with PyTorch included
- **🎯 Zero Configuration** - Works with your existing Python scripts

## 🚀 Quick Start

### Installation

```bash
pip install ml-pipeline-orchestrator
```

### Basic Usage

1. **Create a pipeline config** (`my_pipeline.yml`):

```yaml
name: "ml_training_pipeline"
tasks:
  - name: "fetch_data"
    command: "python scripts/fetch_data.py --output data/"
    
  - name: "train_model"
    command: "python scripts/train.py --data data/ --model model.pkl"
    depends_on: ["fetch_data"]
    retry_count: 3
    
  - name: "evaluate"
    command: "python scripts/evaluate.py --model model.pkl"
    depends_on: ["train_model"]
```

2. **Run your pipeline**:

```bash
# Command line
ml-orchestrator run my_pipeline.yml

# Python API
from ml_orchestrator import Pipeline
pipeline = Pipeline.from_config('my_pipeline.yml')
pipeline.run()
```

3. **Monitor and manage**:

```bash
ml-orchestrator status ml_training_pipeline
ml-orchestrator resume ml_training_pipeline  # Resume if failed
ml-orchestrator list  # Show all pipelines
```

## 🎯 Real-World Example: MNIST Classification

Included is a complete computer vision pipeline that:
- Downloads 70,000 MNIST images
- Trains a PyTorch CNN model  
- Achieves 95%+ accuracy
- Generates comprehensive HTML reports

```bash
# Run the MNIST example
ml-orchestrator run examples/mnist_pipeline.yml
```

## 🔧 Advanced Features

### Environment Variables & Configuration

```yaml
tasks:
  - name: "train"
    command: "python train.py --epochs ${EPOCHS:-10} --lr ${LR:-0.001}"
    environment:
      CUDA_VISIBLE_DEVICES: "0"
      MODEL_TYPE: "cnn"
```

### Error Handling & Retries

```yaml
tasks:
  - name: "flaky_task"
    command: "python unreliable_script.py"
    retry_count: 3
    timeout: 1800
```

### State Management

```python
# Resume failed pipelines
pipeline = Pipeline.from_state('my_pipeline')
pipeline.run(resume=True)

# Check detailed status
status = pipeline.get_status()
print(f"Progress: {status['completed_tasks']}/{status['total_tasks']}")
```

## 💡 Why ML Pipeline Orchestrator?

**vs Apache Airflow:**
- ✅ 5-minute setup vs hours of configuration
- ✅ Lightweight - no web server or database required
- ✅ Built specifically for ML workflows

**vs Manual Scripts:**
- ✅ Automatic dependency resolution
- ✅ Resume failed pipelines without starting over
- ✅ Built-in error handling and logging
- ✅ Professional execution tracking

**vs Other Tools:**
- ✅ No cloud lock-in
- ✅ Works with any Python environment
- ✅ Integrates with existing codebases

## 📋 Configuration Reference

### Task Options

| Option | Description | Example |
|--------|-------------|---------|
| `name` | Unique task identifier | `"preprocess_data"` |
| `command` | Shell command to execute | `"python train.py"` |
| `depends_on` | Task dependencies | `["fetch_data", "validate"]` |
| `retry_count` | Number of retries on failure | `3` |
| `timeout` | Max execution time (seconds) | `3600` |
| `working_dir` | Execution directory | `"/path/to/project"` |
| `environment` | Environment variables | `{"GPU": "0"}` |

### Pipeline Options

```yaml
name: "my_pipeline"
description: "ML training pipeline"
version: "1.0"
author: "Data Team"
tags: ["training", "production"]

tasks:
  # ... task definitions
```

## 🛠️ Integration Examples

### With Jupyter Notebooks

```yaml
tasks:
  - name: "data_analysis"
    command: "jupyter nbconvert --execute analysis.ipynb"
```

### With Docker

```yaml
tasks:
  - name: "containerized_training"
    command: "docker run --rm -v $(pwd):/app my-ml-image python train.py"
```

### With Cloud Services

```yaml
tasks:
  - name: "upload_model"
    command: "aws s3 cp model.pkl s3://my-bucket/models/"
    depends_on: ["train_model"]
```

## 📊 Monitoring & Debugging

### Execution Tracking

```python
# Get detailed execution info
for task_name in pipeline.tasks:
    task_status = pipeline.get_task_status(task_name)
    print(f"{task_name}: {task_status['status']}")
```

### Logging & Debugging

```bash
# Run with debug logging
ml-orchestrator --log-level DEBUG run my_pipeline.yml

# Save logs to file
ml-orchestrator --log-file pipeline.log run my_pipeline.yml
```

## 🎓 Use Cases

- **Model Training Pipelines** - Automate data → preprocess → train → evaluate workflows
- **Data Processing** - Multi-step ETL pipelines with dependency management
- **Model Deployment** - Training → validation → deployment → monitoring chains
- **Batch Inference** - Scheduled prediction pipelines
- **Hyperparameter Tuning** - Coordinate multiple training experiments
- **A/B Testing** - Manage parallel model training and comparison

## 🤝 Contributing

We welcome contributions! The orchestrator is designed to be:
- **Simple** - Easy to understand and extend
- **Reliable** - Thoroughly tested with real ML workloads
- **Practical** - Solves actual problems ML teams face

## 📜 License

MIT License - feel free to use in commercial and open-source projects.

## 🔗 Links

- **Documentation**: [GitHub README](https://github.com/yourusername/ml-orchestrator)
- **Examples**: Complete MNIST pipeline included
- **Issues**: [GitHub Issues](https://github.com/yourusername/ml-orchestrator/issues)
- **PyPI**: [ml-pipeline-orchestrator](https://pypi.org/project/ml-pipeline-orchestrator/)

---

**Built by ML engineers, for ML engineers.** Transform your scripts into production-ready pipelines in minutes, not hours.