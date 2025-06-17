# Sequential-ISNE: Streaming Graph Neural Network Embeddings

A minimal implementation demonstrating Sequential-ISNE with complete data normalization pipeline.

**Based on the original ISNE (Inductive Shallow Node Embedding) method with extensions for streaming chunk processing.**

## Overview

Sequential-ISNE processes documents and code in sequential order, discovering theory-practice bridges between research papers and code implementations. This work extends the original ISNE algorithm to handle streaming data with global sequential IDs and demonstrates a complete data normalization pipeline.

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

## References

This work builds upon the original ISNE (Inductive Shallow Node Embedding) method:

**Original ISNE Paper:**
```bibtex
@inproceedings{yang2020inductive,
  title={Inductive Shallow Node Embedding},
  author={Yang, Yiming and Zhang, Yujun and Liu, Yizhou and Tang, Jian and Wang, Chuan and Li, Juanzi},
  booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management},
  pages={2085--2094},
  year={2020},
  organization={ACM}
}
```

**Original ISNE Code Repository:**
- GitHub: [https://github.com/yzjiao/ISNE](https://github.com/yzjiao/ISNE)

## Acknowledgments

We thank the original ISNE authors for their foundational work on inductive node embeddings. Sequential-ISNE extends their approach to handle streaming document processing and adds comprehensive data normalization capabilities for real-world applications.