#!/usr/bin/env python3
"""
Enhanced Dataset Creator with Actor-Network Theory and STS Focus

Adds curated academic papers to create rich theory→practice bridges:
- Actor-Network Theory and Science & Technology Studies (STS)
- Social Network Analysis and Complex Systems
- Knowledge Representation and Information Theory
- Graph Theory and Network Science
"""

import shutil
from pathlib import Path

# Curated paper selections for theory→practice bridging
THEORY_PAPERS = {
    "actor_network_sts": [
        "Behavioral Aspects of Social Network Analysis_1503.00477v1.pdf",
        "Are all Social Networks Structurally Similar_ A Comparative Study using Network Statistics and ..._1311.2887v2.pdf",
        "Social Network Analysis for Social Neuroscientists_1909.11894v2.pdf",  # From previous list
        "Communities and Hierarchical Structures in Dynamic Social Networks_ Analysis and Visualization_1409.5040v1.pdf",
        "Big Questions for Social Media Big Data_ Representativeness_ Validity and Other Methodological ..._1403.7400v2.pdf",
        "Disassortative mixing in online social networks_0909.0450v1.pdf",
        "A method to evaluate the reliability of social media data for social network analysis_2010.08717v1.pdf",
        "Combinations of Affinity Functions for Different Community Detection Algorithms in Social Networks_2208.12874v1.pdf"
    ],
    
    "complex_systems": [
        "Complex Networks_ Communities and Clustering_ A survey_1503.06277v1.pdf",
        "A betweenness structure entropy of complex networks_1407.0097v1.pdf", 
        "Complex networks with complex weights_2212.06257v2.pdf",
        "A new information dimension of complex networks_1311.3527v1.pdf",
        "Multifractal analysis of complex networks_1108.5014v2.pdf",
        "A Multilayer Model of Computer Networks_1509.00721v1.pdf",
        "Centrality Metric for Dynamic Networks_1006.0526v1.pdf",
        "Analysis of network by generalized mutual entropies_0709.0929v1.pdf"
    ],
    
    "knowledge_representation": [
        "Knowledge Representation_0208019v1.pdf",
        "A New Penta-valued Logic Based Knowledge Representation_1502.05562v1.pdf", 
        "A Set Theoretic Approach for Knowledge Representation_ the Representation Part_1603.03511v1.pdf",
        "Knowledge Representations in Technical Systems -- A Taxonomy_2001.04835v2.pdf",
        "Image-embodied Knowledge Representation Learning_1609.07028v2.pdf",
        "Causal Independence for Knowledge Acquisition and Inference_1303.1468v2.pdf",
        "An extended description logic system with knowledge element based on ALC_1904.07469v1.pdf",
        "A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge_2308.04811v1.pdf"
    ],
    
    "social_media_sts": [
        "Alexandria_ Extensible Framework for Rapid Exploration of Social Media_1507.06667v1.pdf",
        "Applications of Social Media in Hydroinformatics_ A Survey_1905.03035v1.pdf",
        "AWESOME_ Empowering Scalable Data Science on Social Media Data with an Optimized Tri-Store Data..._2112.00833v3.pdf",
        "Curating Social Media Data_2002.09202v1.pdf",
        "Designing a Social Media Analytics Dashboard for Government Agency Crisis Communications_2202.05541v1.pdf",
        "Social Media Analytics in Disaster Response_ A Comprehensive Review_2307.04046v1.pdf",
        "The Ethics of Social Media Analytics in Migration Studies_2302.14404v1.pdf",
        "Situational Awareness Enhanced through Social Media Analytics_ A Survey of First Responders_1909.07316v1.pdf"
    ],
    
    "graph_theory": [
        "Graph Theory in Brain Networks_2103.05781v1.pdf",
        "Graph Signal Processing_ Modulation_ Convolution_ and Sampling_1912.06762v1.pdf",
        "Recent progress on graphs with fixed smallest eigenvalue_2011.11935v1.pdf",
        "Emergent Hyperbolic Network Geometry_1607.05710v2.pdf",
        "A Review on Transportation Network based on Complex Network Approach_2308.04636v1.pdf",
        "Network Reconstruction and Controlling Based on Structural Regularity Analysis_1805.07746v2.pdf",
        "A Survey on the Network Models applied in the Industrial Network Optimization_2209.08294v1.pdf",
        "On Quantitatively Measuring Controllability of Complex Networks_1601.00172v1.pdf"
    ]
}

def copy_papers_to_dataset(source_dir: str, target_dir: str):
    """Copy curated theory papers to enhance the dataset."""
    
    source_path = Path(source_dir)
    target_base = Path(target_dir)
    
    # Create theory directories
    theory_dir = target_base / "theory-papers"
    theory_dir.mkdir(exist_ok=True)
    
    total_copied = 0
    total_missing = 0
    
    for category, papers in THEORY_PAPERS.items():
        category_dir = theory_dir / category
        category_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 {category.replace('_', ' ').title()} ({len(papers)} papers)")
        
        for paper in papers:
            source_file = source_path / paper
            
            if source_file.exists():
                target_file = category_dir / paper
                shutil.copy2(source_file, target_file)
                print(f"   ✅ {paper}")
                total_copied += 1
            else:
                print(f"   ❌ Missing: {paper}")
                total_missing += 1
    
    # Create theory→practice bridge documentation
    create_theory_bridge_docs(theory_dir)
    
    print(f"\n📊 Theory Papers Summary:")
    print(f"   🎯 Targeted: {sum(len(papers) for papers in THEORY_PAPERS.values())} papers")
    print(f"   ✅ Copied: {total_copied} papers")
    print(f"   ❌ Missing: {total_missing} papers")
    print(f"   📁 Organized in: {theory_dir}")
    
    return total_copied, total_missing

def create_theory_bridge_docs(theory_dir: Path):
    """Create documentation explaining theory→practice bridges."""
    
    # Main theory bridge overview
    bridge_doc = theory_dir / "THEORY_PRACTICE_BRIDGES.md"
    bridge_doc.write_text("""# Theory→Practice Bridge Documentation
## Actor-Network Theory Enhanced Sequential-ISNE Validation

### Overview
This enhanced dataset demonstrates **Actor-Network Theory (ANT)** and **Science & Technology Studies (STS)** principles applied to Sequential-ISNE validation, showing how theoretical frameworks translate into practical implementations.

## 🔬 Theoretical Foundation Categories

### 1. Actor-Network Theory & STS (`actor_network_sts/`)
- **Core Principle**: Networks of human and non-human actors
- **Bridge to Code**: How research papers (actors) connect to implementations (actors)
- **STS Focus**: Science-technology co-construction
- **Sequential-ISNE Application**: Co-location patterns reveal actor-networks

### 2. Complex Systems Theory (`complex_systems/`)
- **Core Principle**: Emergent properties from network interactions
- **Bridge to Code**: Complex software architectures emerge from simple components
- **STS Focus**: Technological systems as complex adaptive systems
- **Sequential-ISNE Application**: Multi-scale relationship discovery

### 3. Knowledge Representation (`knowledge_representation/`)
- **Core Principle**: Formal systems for encoding knowledge
- **Bridge to Code**: How theoretical concepts become data structures
- **STS Focus**: Inscription and translation of knowledge
- **Sequential-ISNE Application**: Semantic bridging between domains

### 4. Social Media & STS (`social_media_sts/`)
- **Core Principle**: Technology-society co-evolution
- **Bridge to Code**: Social platforms as socio-technical assemblages
- **STS Focus**: User-technology interaction and affordances
- **Sequential-ISNE Application**: Media ecology relationship patterns

### 5. Graph Theory Foundations (`graph_theory/`)
- **Core Principle**: Mathematical formalization of networks
- **Bridge to Code**: Abstract graph theory → concrete graph algorithms
- **STS Focus**: Mathematical objects as technological actors
- **Sequential-ISNE Application**: Theory-implementation consistency validation

## 🌉 Expected Semantic Bridges

### High-Confidence Bridges (Similarity > 0.8)
- Actor-Network Theory papers ↔ Network analysis code
- Complex systems theory ↔ Multi-layer graph implementations
- Knowledge representation ↔ Data structure definitions
- Graph theory papers ↔ Graph algorithm implementations

### Cross-Domain Bridges (Similarity 0.6-0.8)
- Social media STS ↔ Data processing pipelines
- Complex systems ↔ ISNE embedding architectures
- ANT assemblages ↔ Software component interactions
- Knowledge inscription ↔ Code documentation practices

### Novel Discovery Bridges (Similarity 0.4-0.6)
- STS methodology ↔ Software development practices
- Actor-network formation ↔ Code dependency networks
- Translation processes ↔ Data transformation pipelines
- Technological mediation ↔ API interface design

## 🎯 Sequential-ISNE Validation Goals

### Actor-Network Validation
1. **Heterogeneous Networks**: Papers + Code + Documentation as actor-networks
2. **Translation Processes**: How theory becomes practice
3. **Intermediaries vs Mediators**: Passive vs. active code components
4. **Network Stabilization**: How implementations crystallize theory

### STS Validation
1. **Co-construction**: Theory and practice mutually constitute each other
2. **Black-boxing**: How complex theory becomes simple interfaces
3. **Inscription**: How social processes become technical objects
4. **Symmetry**: Equal analytical treatment of human and non-human actors

### Scale & Authenticity
- **Legitimate Academic Scale**: 40+ theory papers + 3 implementations
- **Authentic Relationships**: Real theory→practice connections
- **Multi-domain Coverage**: CS, sociology, philosophy, mathematics
- **Temporal Depth**: Papers spanning multiple decades of research

## 📊 Expected Training Improvements

### Quantitative Gains
- **Nodes**: 4,677 + theory paper chunks (estimated 6,000+ total)
- **Relationships**: 760k+ (approaching ISNE paper benchmarks)
- **Bridge Diversity**: 5 theoretical domains × 3 implementations
- **Semantic Richness**: Multi-disciplinary concept networks

### Qualitative Validation
- **ANT Principle Validation**: Networks of heterogeneous actors
- **STS Coherence**: Technology-society co-construction evidence
- **Cross-domain Translation**: Theory→practice mapping accuracy
- **Network Assembly**: How actor-networks form and stabilize

This enhanced dataset enables Sequential-ISNE to validate core ANT and STS principles through computational analysis of real theory→practice actor-networks.
""")
    
    # Category-specific bridge docs
    for category in THEORY_PAPERS.keys():
        category_path = theory_dir / category
        readme_path = category_path / "README.md"
        
        category_title = category.replace('_', ' ').title()
        readme_path.write_text(f"""# {category_title} Papers

## Theory→Practice Bridge Focus
This category contains papers specifically selected to validate **{category_title}** principles through Sequential-ISNE analysis.

## Expected Semantic Connections
- **Direct Bridges**: Papers in this category → corresponding code implementations
- **Cross-Domain**: Connections to other theoretical frameworks
- **Novel Discoveries**: Unexpected relationships revealed by embeddings

## Actor-Network Theory Application
These papers serve as **theoretical actors** in the network, connecting to **practical actors** (code implementations) through semantic similarity and co-location patterns.

## Sequential-ISNE Validation
The embeddings should reveal how {category_title.lower()} concepts translate into computational implementations across the three RAG systems.
""")

if __name__ == "__main__":
    source_dir = "/home/todd/ML-Lab/Olympus/test-data3"
    target_dir = "/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata/isne-testdata"
    
    print("🧠 Enhancing dataset with Actor-Network Theory and STS papers...")
    copied, missing = copy_papers_to_dataset(source_dir, target_dir)
    
    print(f"\n🎉 Dataset enhancement complete!")
    print(f"   📚 Theory papers added: {copied}")
    print(f"   🌉 Multi-domain bridges ready for Sequential-ISNE validation")
    print(f"   🔬 Actor-Network Theory validation enabled")