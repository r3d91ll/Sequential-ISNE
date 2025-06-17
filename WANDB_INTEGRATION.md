# Weights & Biases Integration Guide

Sequential-ISNE includes comprehensive Weights & Biases (wandb) integration for experiment tracking, metrics logging, and reproducible research.

## 🚀 Quick Setup

1. **Get your wandb API key:**
   ```bash
   # Visit https://wandb.ai/authorize and copy your API key
   ```

2. **Configure environment:**
   ```bash
   cp .env.template .env
   # Edit .env and replace 'your_wandb_api_key_here' with your actual API key
   ```

3. **Run training with wandb logging:**
   ```bash
   python train_and_log_model.py
   ```

## 📊 What Gets Logged

### Experiment Configuration
- Algorithm: Sequential-ISNE
- Processing strategy: doc_first_depth
- Chunk size and overlap settings
- Embedding dimensions
- Dataset information

### Training Metrics
- **Data Preparation**: File counts, repository stats, file type distribution
- **Document Processing**: Success rates, chunk creation, text extraction metrics
- **Hierarchical Processing**: Strategy performance, chunk distribution, processing times
- **Relationship Discovery**: Total relationships, cross-document connections, theory→practice bridges
- **Model Training**: Training epochs, loss values, embedding success rates
- **Model Evaluation**: Quality metrics, validation scores, capability assessments

### Research Validation
- Ship of Theseus principle validation
- Hierarchical processing effectiveness
- Theory→practice bridging detection
- Academic repository processing success

## 🔧 Configuration Options

### Environment Variables

```bash
# Required for wandb logging
WANDB_API_KEY=your_actual_api_key_here

# Optional customizations
WANDB_ENTITY=your_username_or_team_name
WANDB_PROJECT=sequential-isne-research
WANDB_MODE=online  # or offline, disabled
```

### Project Organization

- **Project**: `sequential-isne-research` (configurable)
- **Experiment naming**: Automatic timestamps (`full-pipeline-YYYYMMDD-HHMMSS`)
- **Tags**: Automatically tagged with algorithm type and dataset

## 📈 Viewing Results

### Wandb Dashboard
Once training completes, visit your wandb dashboard:
```
https://wandb.ai/[your-entity]/sequential-isne-research
```

### Local Results
Training also saves local artifacts:
```
experiments/trained_models/
├── sequential_isne_[experiment-name].json     # Model configuration
└── training_results_[experiment-name].json   # Complete metrics
```

## 🎯 Research Benefits

### Reproducibility
- All hyperparameters and configurations logged
- Environment information captured
- Random seeds and model versions tracked

### Collaboration
- Share experiment results with team members
- Compare different processing strategies
- Track model performance over time

### Academic Publication
- Complete experimental records for paper citations
- Quantitative metrics for validation claims
- Reproducible results for peer review

## 🛠️ Advanced Usage

### Custom Experiments
```python
from train_and_log_model import SequentialISNETrainer

trainer = SequentialISNETrainer(
    project_name="my-sequential-isne-experiments",
    experiment_name="custom-strategy-test",
    log_to_wandb=True
)

results = trainer.run_complete_training()
```

### Offline Mode
```bash
export WANDB_MODE=offline
python train_and_log_model.py
# Later: wandb sync wandb/offline-run-*/
```

### Disable Wandb
```bash
export WANDB_MODE=disabled
python train_and_log_model.py
# Runs without any wandb logging
```

## 📋 Troubleshooting

### API Key Issues
```bash
# Verify API key is set correctly
echo $WANDB_API_KEY

# Login manually if needed
wandb login
```

### Connection Problems
```bash
# Use offline mode if connection issues
export WANDB_MODE=offline
python train_and_log_model.py
```

### Missing Dependencies
```bash
# Install wandb if missing
pip install wandb
# or with Poetry
poetry add wandb
```

## 🎓 Academic Workflow

For academic research, we recommend:

1. **Create dedicated projects** for each paper/study
2. **Use descriptive experiment names** that reflect the research question
3. **Tag experiments** with relevant metadata (hypothesis, dataset, configuration)
4. **Export metrics** for inclusion in papers and presentations
5. **Share project links** with collaborators and reviewers

### Example Academic Workflow
```bash
# Set up research-specific project
export WANDB_PROJECT=sequential-isne-paper-2025
export WANDB_ENTITY=university-research-lab

# Run experiments for paper
python train_and_log_model.py  # Baseline
# Modify configuration for ablation studies
python train_and_log_model.py  # Strategy comparison
python train_and_log_model.py  # Scale analysis

# Export results for paper
wandb export sequential-isne-paper-2025
```

This integration ensures that Sequential-ISNE research is fully reproducible and meets academic standards for experimental validation and reporting.