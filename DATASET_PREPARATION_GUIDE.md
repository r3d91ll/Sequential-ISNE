# Dataset Preparation Guide for Directory-Informed ISNE

## Overview

This guide provides step-by-step instructions for preparing datasets that maximize ISNE bootstrap effectiveness through proper documentation cross-reference structure.

***NOTE: this methodology is intended for code bases and was tested extensivly on Python codebases***

## 🎯 Core Principle

***Documentation Cross-References = Explicit Graph Edges***

Just as Python imports create explicit dependencies between modules, markdown cross-references create explicit semantic relationships between directories and abstraction levels.

## 📋 Step-by-Step Preparation Process

### Step 1: Audit Existing Documentation

**Scan your dataset for existing documentation:**

```bash
find /path/to/dataset -name "*.md" -o -name "*.txt" | head -20
```

**Identify gaps:**

- Directories without README.md files
- Documentation without cross-references
- Isolated theory papers or implementations

### Step 2: Create Directory-Level Documentation

**For each directory, create a README.md that includes:**

#### Template for Implementation Directories

```markdown
# [Directory Name]

## Purpose
Brief description of what this directory contains and its role in the project.

## Key Components
- [file1.py](./file1.py) - Primary algorithm implementation
- [file2.py](./file2.py) - Supporting utilities
- [config.json](./config.json) - Configuration settings

## Theoretical Foundation
This implementation is based on:
- [Theory Paper Title](../theory-papers/category/paper.pdf)
- [Related Concept](../theory-papers/related-category/README.md)

## Related Implementations
- [Complementary Approach](../other-implementation/README.md)
- [Alternative Method](../alternative-approach/README.md)

## Integration Points
- Used by: [Higher-level System](../main-system/README.md)
- Depends on: [Lower-level Utilities](../utilities/README.md)
```

#### Template for Theory Paper Directories

```markdown
# [Theory Category Name]

## Overview
Description of the theoretical domain and its relevance.

## Papers in this Category
- [Paper 1](./paper1.pdf) - Key contribution: [brief description]
- [Paper 2](./paper2.pdf) - Key contribution: [brief description]
- [Paper 3](./paper3.pdf) - Key contribution: [brief description]

## Practical Implementations
These theoretical concepts are implemented in:
- [Implementation A](../../code-directory-a/README.md) - [specific mapping]
- [Implementation B](../../code-directory-b/README.md) - [specific mapping]

## Cross-Domain Connections
- Related to [Other Theory Category](../other-category/README.md)
- Foundational for [Applied Category](../applied-category/README.md)

## Key Concepts for ISNE
- **Primary Concept**: How it applies to graph embedding
- **Secondary Concept**: Relevance to directory-informed bootstrap
```

### Step 3: Create Top-Level Overview Documentation

**Main dataset README.md should include:**

```markdown
# Dataset Name

## Overview
[Description of dataset purpose and scope]

## Directory Structure

### Implementation Repositories
- [Repository A](./repo-a/README.md) - [brief description]
- [Repository B](./repo-b/README.md) - [brief description]
- [Repository C](./repo-c/README.md) - [brief description]

### Theoretical Foundation
- [Theory Overview](./theory-papers/README.md) - Complete theoretical framework
- [Category 1](./theory-papers/category-1/README.md) - [specific domain]
- [Category 2](./theory-papers/category-2/README.md) - [specific domain]

### Cross-Repository Connections
- **Repository A** ↔ [Theory Category 1](./theory-papers/category-1/README.md) - [relationship]
- **Repository B** ↔ [Theory Category 2](./theory-papers/category-2/README.md) - [relationship]
- **Repository A** ↔ **Repository B** - [shared concepts/approaches]

## Theory-Practice Bridge Map
| Theory Paper | Implementation | Key Bridge |
|-------------|----------------|------------|
| [Paper 1](./theory-papers/cat1/paper1.pdf) | [Code A](./repo-a/main.py) | Algorithm X |
| [Paper 2](./theory-papers/cat2/paper2.pdf) | [Code B](./repo-b/core.py) | Method Y |

## Dataset Validation
- [x] All directories have README.md
- [x] Cross-references link related components
- [x] Theory-practice bridges documented
- [x] Hierarchical documentation structure
```

### Step 4: Implement Cross-Reference Links

**Use relative paths for robustness:**

```markdown
# Good - Relative paths
[Related Implementation](../other-dir/README.md)
[Theory Paper](../../theory-papers/category/README.md)

# Avoid - Absolute paths
[Bad Example](/full/path/to/file.md)
```

**Create bidirectional references:**

- Theory papers reference implementations
- Implementations reference theory papers
- Related implementations reference each other
- Subdirectories reference parent/sibling directories

### Step 5: Remove Irrelevant Content

**Identify and remove files that pollute the semantic space:**

```bash
# Find potential boilerplate files
find /path/to/dataset -name "SECURITY.md" -o -name "CODE_OF_CONDUCT.md" -o -name "CONTRIBUTING.md"

# Review and remove if they're generic boilerplate
rm dataset/SECURITY.md dataset/CODE_OF_CONDUCT.md
```

**Keep only files that contribute to research understanding:**

- Theory papers (PDFs)
- Implementation code (.py, .js, etc.)
- Custom documentation (README.md with cross-references)
- Configuration files relevant to understanding the system

### Step 6: Validate Cross-Reference Structure

**Test the documentation network:**

```bash
# Check that all referenced files exist
python validate_documentation_links.py /path/to/dataset

# Or manually verify key cross-references
grep -r "\[.*\](\..*\.md)" /path/to/dataset
```

**Verification checklist:**

- [ ] Every directory has custom README.md
- [ ] Top-level documentation links to all major subdirectories
- [ ] Subdirectories reference related directories
- [ ] Theory papers explicitly linked to implementations
- [ ] Cross-domain relationships documented
- [ ] No broken links in cross-references
- [ ] Removed irrelevant boilerplate files

## 🔬 Expected ISNE Bootstrap Improvements

### Graph Edge Types Created

1. **Co-location edges** (automatic): Files in same directory
2. **Import edges** (automatic): Python imports between files
3. **Documentation edges** (explicit): Markdown cross-references
4. **Semantic edges** (learned): ISNE-discovered relationships

### Validation Results

- **Before proper documentation**: Limited to co-location + imports
- **After cross-references**: 2-3x more initial edges for ISNE training
- **Enhanced discovery**: Better theory-practice bridge detection

## 🛠 Tools and Scripts

### Documentation Link Validator (Optional)

```python
#!/usr/bin/env python3
"""Validate documentation cross-references in dataset."""

import re
from pathlib import Path

def validate_links(dataset_path):
    for md_file in Path(dataset_path).rglob("*.md"):
        content = md_file.read_text()
        links = re.findall(r'\[.*?\]\((\..*?\.md)\)', content)
        
        for link in links:
            target = (md_file.parent / link).resolve()
            if not target.exists():
                print(f"Broken link in {md_file}: {link}")

if __name__ == "__main__":
    validate_links("/path/to/dataset")
```

## ✅ Success Criteria

A properly prepared dataset will show:

- **Rich initial graph structure** with multiple edge types
- **Strong theory-practice bridge detection** by ISNE
- **Cross-modal relationships** between documentation, theory, and code
- **Hierarchical semantic organization** reflecting conceptual structure

The investment in documentation cross-references pays off through significantly improved ISNE relationship discovery and more meaningful theory-practice bridge detection.
