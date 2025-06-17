# Sequential-ISNE Setup Guide

## Environment Setup

### 1. Install Dependencies

```bash
# Basic installation
poetry install

# With GPU support (PyTorch, sentence-transformers)
poetry install --extras gpu

# With document processing (Docling for PDFs)
poetry install --extras docs

# Full installation (all features)
poetry install --extras full
```

### 2. Configure Weights & Biases (Optional)

For experiment tracking and metrics logging:

1. **Copy environment template:**
   ```bash
   cp .env.template .env
   ```

2. **Get your wandb API key:**
   - Visit https://wandb.ai/authorize
   - Copy your API key

3. **Edit `.env` file:**
   ```bash
   # Sequential-ISNE Environment Configuration
   WANDB_API_KEY=your_actual_api_key_here
   WANDB_ENTITY=your_username_or_team  # Optional
   WANDB_PROJECT=sequential-isne-research  # Optional
   WANDB_MODE=online  # or offline, disabled
   ```

4. **Test wandb connection:**
   ```bash
   poetry run python -c "import wandb; wandb.login()"
   ```

### 3. Run Training Pipeline

```bash
# Train Sequential-ISNE model with wandb logging
poetry run python train_and_log_model.py

# Or without Poetry
cd Sequential-ISNE
python train_and_log_model.py
```

### 4. View Results

- **Local results:** Check `experiments/trained_models/`
- **Wandb dashboard:** Visit your project at https://wandb.ai/[entity]/[project]

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `WANDB_API_KEY` | Weights & Biases API key | None | For wandb logging |
| `WANDB_ENTITY` | Username or team name | None | Optional |
| `WANDB_PROJECT` | Project name | `sequential-isne-research` | Optional |
| `WANDB_MODE` | Logging mode | `online` | Optional |

## Troubleshooting

### Missing Dependencies
```bash
# If you see "wandb not available"
poetry add wandb

# If you see "python-dotenv not available" 
poetry add python-dotenv

# If you see "docling not available"
poetry install --extras docs
```

### Wandb Issues
```bash
# Login manually
poetry run wandb login

# Run in offline mode
export WANDB_MODE=offline
poetry run python train_and_log_model.py

# Disable wandb completely
export WANDB_MODE=disabled
poetry run python train_and_log_model.py
```

### PDF Processing Issues
```bash
# Install document processing dependencies
poetry install --extras docs

# Or use text/markdown files only (PDFs will be skipped)
# The pipeline gracefully handles missing PDF processors
```