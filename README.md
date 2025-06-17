# Sequential-ISNE: Directory-Informed Graph Neural Network Embeddings

A directory-informed implementation of ISNE (Inductive Shallow Node Embedding) that bootstraps graph structure from filesystem organization.

**Based on the original ISNE method with directory-informed graph bootstrap for proper ISNE application.**

## Overview

Sequential-ISNE discovers theory-practice bridges between research papers and code implementations by using directory structure as an implicit knowledge graph. This approach solves the fundamental challenge of applying ISNE (a graph embedding algorithm) by bootstrapping proper graph structure from filesystem organization, creating semantically meaningful relationships between co-located files, import dependencies, and cross-modal content.

## Quick Start

```bash
# Install dependencies
poetry install

# Run focused demo
python3 simple_demo.py /path/to/dataset

# Run comprehensive academic validation
python3 academic_test.py

# Full pipeline (for well-prepared datasets)
python3 demo.py /path/to/dataset --config config.yaml
```

**⚠️ Important**: For optimal results, datasets must be properly prepared with documentation cross-references. See [Dataset Preparation Guide](./DATASET_PREPARATION_GUIDE.md) for detailed instructions.

## Key Features

### Directory-Informed Graph Bootstrap
- **DirectoryGraph Class**: Builds initial graph from filesystem structure
- **Co-location Edges**: Files in same directory have strong relationships (weight 1.0)
- **Import Edges**: Python import statements create directed edges (weight 0.9)  
- **Semantic Edges**: Cross-modal embeddings create theory-practice bridges (threshold 0.7)

### Proper ISNE Implementation
- **Graph-Based Training**: ISNE operates on actual graph neighborhoods (not sequential chunks)
- **Chunk-to-Node Mapping**: Resolves the fundamental chunk-to-node mapping problem
- **Cross-Modal Learning**: Discovers relationships between documentation and code

### Complete Pipeline
- **Data Normalization** with Docling (PDFs → markdown)
- **Unified Chunking** with Chonky (text) and AST (Python)  
- **CodeBERT Embeddings** for both text and code modalities
- **Directory-Informed Bridge Detection** using graph structure

## Architecture Recovery

This implementation recovers from a fundamental conceptual error where ISNE (a graph embedding algorithm) was incorrectly applied to sequential chunks without proper graph structure. The solution uses directory structure as an implicit knowledge graph:

```
Directory Structure → Implicit Graph → Proper ISNE Training
     ↓                      ↓                    ↓
Co-located files    →  Graph edges    →  Neighborhood aggregation
Import statements   →  Directed edges  →  Structural relationships  
Semantic similarity →  Cross-modal     →  Theory-practice bridges
```

## Latest Results

**Academic Dataset Test (101 documents, 588 chunks)**:
- **📊 Directory Graph**: 715 nodes, 203,886 edges  
- **🌉 Theory-Practice Bridges**: 34,600 detected
- **🎯 Training**: 9,900 training pairs, 5 epochs
- **📉 Final Loss**: 0.000001 (excellent convergence)
- **🔬 Graph Stats**: 28.7% density, 17.0% clustering coefficient

**Previous Sequential Results** (for comparison):
- 3,260 chunks with 416,357 training pairs
- 773 bridges detected using flawed sequential approach
- 97.4% clustering coefficient, 0.000111 final loss

## Dependencies

See `requirements.txt` or use Poetry with `pyproject.toml`

## Core Components

### DirectoryGraph (`directory_graph.py`)
Bootstrap graph construction from filesystem structure:
```python
from directory_graph import DirectoryGraph

# Create graph from directory structure
graph = DirectoryGraph()
graph.bootstrap_from_directory(root_path, file_contents)
graph.extend_with_semantic_similarity(embeddings, threshold=0.7)

# Find theory-practice bridges
bridges = graph.find_theory_practice_bridges()
```

### Enhanced Sequential-ISNE (`src/sequential_isne.py`)
Proper ISNE training with directory-informed graphs:
```python
from src.sequential_isne import SequentialISNE

# Train with directory structure
model = SequentialISNE(config)
model.build_graph_from_directory_graph(directory_graph, chunks)
results = model.train_embeddings()
```

## Key Innovation

**Directory Structure as Implicit Knowledge Graph**: The breakthrough insight is using filesystem organization to bootstrap the graph structure that ISNE requires. This solves the chicken-and-egg problem of needing a graph to apply a graph embedding algorithm.

## 📋 Dataset Preparation Requirements

### Critical: Documentation Cross-Reference Structure

For optimal ISNE bootstrap, datasets **must** include:

#### 1. **Directory-Level Documentation**
Every directory requires custom documentation:
```
dataset/
├── README.md                    # Top-level overview with cross-references
├── subdirectory-a/
│   └── README.md               # Directory-specific documentation
├── subdirectory-b/
│   └── README.md               # Directory-specific documentation
└── theory-papers/
    ├── README.md               # Theory overview
    ├── category-1/
    │   └── README.md           # Category-specific documentation
    └── category-2/
        └── README.md           # Category-specific documentation
```

#### 2. **Explicit Cross-References**
Documentation files must link to each other (creating "documentation imports"):

**Top-level README.md:**
```markdown
## Documentation Structure
- [Implementation A](./subdirectory-a/README.md) - Core algorithms
- [Implementation B](./subdirectory-b/README.md) - Supporting utilities  
- [Theory Papers](./theory-papers/README.md) - Theoretical foundations

## Cross-Domain Connections
- Implementation A ↔ [Theory Category 1](./theory-papers/category-1/README.md)
- Implementation B ↔ [Theory Category 2](./theory-papers/category-2/README.md)
```

**Subdirectory README.md:**
```markdown
# Implementation A

## Theoretical Foundation
Based on concepts from [Theory Category 1](../theory-papers/category-1/README.md).

## Related Implementations
- [Implementation B](../subdirectory-b/README.md) - Complementary approach
- [Top-level Overview](../README.md) - Complete system context
```

#### 3. **Theory-Practice Bridge Documentation**
Explicit mapping between theoretical papers and code implementations:
```markdown
## Theory → Code Mapping
- **Paper Section 2.1** → [algorithm.py](./src/algorithm.py)
- **Theorem 3** → [optimization.py](./src/optimization.py)
- **Figure 4** → [visualization.py](./utils/visualization.py)
```

#### 4. **Why This Matters for ISNE Bootstrap**

Documentation cross-references create **explicit graph edges**:
- **Co-location edges**: Files in same directory (implicit)
- **Import edges**: Python import statements (explicit)  
- **Documentation edges**: Markdown cross-references (explicit)
- **Semantic edges**: ISNE-discovered relationships (learned)

Without proper documentation cross-references, ISNE can only rely on co-location and imports, missing crucial **conceptual relationships** between directories and abstraction levels.

### 🔧 **Dataset Validation Checklist**

Before running ISNE bootstrap:
- [ ] Every directory has custom README.md
- [ ] Top-level documentation links to all subdirectories  
- [ ] Subdirectory documentation references related directories
- [ ] Theory papers are explicitly linked to implementations
- [ ] Cross-domain relationships are documented
- [ ] Remove irrelevant boilerplate (SECURITY.md, etc.)

**Result**: Rich graph structure with multiple edge types enabling superior ISNE relationship discovery.

## References

This work builds upon the original ISNE (Inductive Shallow Node Embedding) method:

**Original ISNE Paper:**
```bibtex
@article{kiss2024unsupervised,
  title={Unsupervised Graph Representation Learning with Inductive Shallow Node Embedding},
  author={Kiss, Richárd and Szűcs, Gábor},
  journal={Complex \& Intelligent Systems},
  pages={7333--7348},
  year={2024},
  publisher={Springer}
}
```

## Acknowledgments

We thank the original ISNE authors (Kiss & Szűcs) for their foundational work on inductive node embeddings. This directory-informed extension demonstrates how filesystem structure can serve as an implicit knowledge graph for bootstrapping proper ISNE training, enabling cross-modal theory-practice bridge discovery in academic research datasets.