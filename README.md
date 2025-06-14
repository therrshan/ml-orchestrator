# ML Pipeline Orchestrator

A lightweight, ML pipeline orchestrator that transforms your individual ML scripts into reliable, automated workflows. Built specifically for robust Machine Learning pipeline management without the complexity of cloud systems.

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

## 🎓 Use Cases

- **Model Training Pipelines** - Automate data → preprocess → train → evaluate workflows
- **Data Processing** - Multi-step ETL pipelines with dependency management
- **Model Deployment** - Training → validation → deployment → monitoring chains
- **Batch Inference** - Scheduled prediction pipelines
- **Hyperparameter Tuning** - Coordinate multiple training experiments
- **A/B Testing** - Manage parallel model training and comparison

## 📜 License

MIT License - feel free to use in commercial and open-source projects.

## 🔗 Links

- **PyPI**: [ml-pipeline-orchestrator](https://pypi.org/project/ml-pipeline-orchestrator/)

---
