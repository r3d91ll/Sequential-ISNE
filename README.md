# Sequential-ISNE: Streaming Graph Neural Network Embeddings

A minimal implementation demonstrating Sequential-ISNE with complete data normalization pipeline.

## Overview

Sequential-ISNE processes documents and code in sequential order, discovering theory-practice bridges between research papers and code implementations.

## Quick Start

```bash
# Install dependencies
poetry install

# Run demo
poetry run python demo.py /path/to/dataset

# With configuration
poetry run python demo.py /path/to/dataset --config config.yaml
```

## Key Features

- **Data Normalization** with Docling (PDFs → markdown)
- **Unified Chunking** with Chonky (text) and AST (Python)  
- **CodeBERT Embeddings** for both modalities
- **Theory-Practice Bridge Detection** (773 bridges found!)
- **Sequential-ISNE Training** at academic scale

## Results

Validated on 3,260 chunks with 416,357 training pairs:
- **773 theory-practice bridges** detected
- **97.4% clustering coefficient**
- **0.000111 final loss**

## Dependencies

See `requirements.txt` or use Poetry with `pyproject.toml`