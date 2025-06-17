#!/usr/bin/env python3
"""
Sequential-ISNE: NetworkX-based ISNE Model for Academic Research

Implements the core Sequential-ISNE model using NetworkX graphs for simplicity
and reproducibility. This is the academic research implementation focused on
validating the streaming chunk processing approach.

Key features:
- NetworkX graph construction from streaming relationships
- Simple neural network for learning embeddings
- Direct chunk_id to node mapping (solving the fundamental mapping problem)
- Comprehensive evaluation and benchmarking capabilities
"""

import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from collections import defaultdict

# Optional imports for neural network
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for Sequential-ISNE training."""
    embedding_dim: int = 384
    hidden_dim: int = 256
    num_layers: int = 2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    negative_samples: int = 5
    device: str = "auto"


@dataclass
class RelationshipEdge:
    """Represents a relationship edge in the graph."""
    from_chunk_id: int
    to_chunk_id: int
    relationship_type: str
    confidence: float
    context: str


class SequentialGraph:
    """
    NetworkX-based graph for Sequential-ISNE training.
    
    Manages the graph structure built from streaming chunk relationships,
    providing the foundation for ISNE embedding learning.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for relationship types
        self.chunk_metadata: Dict[int, Dict[str, Any]] = {}
        self.relationship_types: Set[str] = set()
        
        logger.info("Initialized Sequential-ISNE graph")
    
    def add_chunks(self, chunks: List['StreamingChunk']) -> None:
        """Add chunks as nodes to the graph."""
        for chunk in chunks:
            # Only add content chunks as nodes (not boundary markers)
            if chunk.metadata.chunk_type == "content":
                self.graph.add_node(
                    chunk.chunk_id,
                    content=chunk.content[:100],  # Truncated for memory
                    doc_path=chunk.metadata.doc_path,
                    directory=chunk.metadata.directory,
                    chunk_type=chunk.metadata.chunk_type,
                    processing_order=chunk.metadata.processing_order
                )
                
                # Store full metadata separately
                self.chunk_metadata[chunk.chunk_id] = {
                    'doc_path': chunk.metadata.doc_path,
                    'directory': chunk.metadata.directory,
                    'file_extension': chunk.metadata.file_extension,
                    'processing_order': chunk.metadata.processing_order,
                    'content_length': len(chunk.content)
                }
        
        logger.info(f"Added {len([c for c in chunks if c.metadata.chunk_type == 'content'])} chunks as graph nodes")
    
    def add_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        """Add relationships as edges to the graph."""
        edges_added = 0
        
        for rel in relationships:
            from_id = rel['from_chunk_id']
            to_id = rel['to_chunk_id']
            rel_type = rel['relationship_type']
            confidence = rel['confidence']
            
            # Only add edges between content chunks
            if from_id in self.graph.nodes and to_id in self.graph.nodes:
                self.graph.add_edge(
                    from_id,
                    to_id,
                    relationship_type=rel_type,
                    confidence=confidence,
                    context=rel.get('context', '')
                )
                
                self.relationship_types.add(rel_type)
                edges_added += 1
        
        logger.info(f"Added {edges_added} relationship edges to graph")
        logger.info(f"Relationship types: {self.relationship_types}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        if not self.graph.nodes:
            return {"error": "Empty graph"}
        
        # Basic graph metrics
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "relationship_types": list(self.relationship_types),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph)
        }
        
        # Relationship type distribution
        rel_type_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            rel_type_counts[data['relationship_type']] += 1
        stats["relationship_distribution"] = dict(rel_type_counts)
        
        # Directory co-location analysis
        directory_stats = defaultdict(lambda: {"nodes": 0, "internal_edges": 0})
        for node_id, data in self.graph.nodes(data=True):
            directory = data.get('directory', 'unknown')
            directory_stats[directory]["nodes"] += 1
        
        for from_id, to_id, edge_data in self.graph.edges(data=True):
            from_dir = self.graph.nodes[from_id].get('directory', 'unknown')
            to_dir = self.graph.nodes[to_id].get('directory', 'unknown')
            if from_dir == to_dir:
                directory_stats[from_dir]["internal_edges"] += 1
        
        stats["directory_analysis"] = dict(directory_stats)
        
        return stats
    
    def get_node_neighbors(self, node_id: int, max_neighbors: int = 10) -> List[Tuple[int, float]]:
        """Get neighbors of a node with relationship confidences."""
        if node_id not in self.graph.nodes:
            return []
        
        neighbors = []
        
        # Outgoing edges
        for neighbor_id in self.graph.successors(node_id):
            edge_data = self.graph.edges[node_id, neighbor_id]
            confidence = edge_data.get('confidence', 0.5)
            neighbors.append((neighbor_id, confidence))
        
        # Incoming edges
        for neighbor_id in self.graph.predecessors(node_id):
            edge_data = self.graph.edges[neighbor_id, node_id]
            confidence = edge_data.get('confidence', 0.5)
            neighbors.append((neighbor_id, confidence))
        
        # Sort by confidence and limit
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:max_neighbors]
    
    def save_graph(self, output_path: str) -> None:
        """Save the graph and metadata."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save graph structure
        graph_file = output_file.with_suffix('.graphml')
        nx.write_graphml(self.graph, graph_file)
        
        # Save metadata
        metadata_file = output_file.with_suffix('.metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'chunk_metadata': self.chunk_metadata,
                'relationship_types': list(self.relationship_types),
                'statistics': self.get_graph_statistics()
            }, f, indent=2)
        
        logger.info(f"Saved graph to {graph_file} and metadata to {metadata_file}")


if HAS_TORCH:
    class SimpleISNEModel(nn.Module):
        """
        Simple ISNE model for learning embeddings from graph structure.
        
        This is a simplified version focused on academic research and validation
        rather than production performance.
        """
        
        def __init__(self, config: TrainingConfig, num_nodes: int = 10000):
            super().__init__()
            self.config = config
            self.num_nodes = max(num_nodes, 15000)  # Ensure sufficient capacity for academic scale
            
            # Dynamic embedding layers based on actual node count
            self.node_embedding = nn.Embedding(self.num_nodes, config.embedding_dim)
            self.hidden = nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.hidden_dim, config.embedding_dim)
            )
            
            self.init_weights()
        
        def init_weights(self):
            """Initialize model weights."""
            nn.init.xavier_uniform_(self.node_embedding.weight)
            for layer in self.hidden:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
        
        def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
            """Forward pass: node_ids -> embeddings."""
            # Clamp node_ids to valid range to prevent index errors
            node_ids = torch.clamp(node_ids, 0, self.num_nodes - 1)
            embeddings = self.node_embedding(node_ids)
            return self.hidden(embeddings)
else:
    # Fallback for when torch is not available
    class SimpleISNEModel:
        pass


class SequentialISNE:
    """
    Main Sequential-ISNE implementation for academic research.
    
    Combines graph construction, training, and evaluation in a simple,
    reproducible package suitable for research and benchmarking.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.graph = SequentialGraph()
        self.model = None
        self.node_to_index: Dict[int, int] = {}  # chunk_id -> model_index mapping
        self.index_to_node: Dict[int, int] = {}  # model_index -> chunk_id mapping
        self.trained_embeddings: Optional[np.ndarray] = None
        
        logger.info(f"Initialized Sequential-ISNE with config: {self.config}")
    
    def build_graph_from_chunks(
        self, 
        chunks: List['StreamingChunk'], 
        relationships: List[Dict[str, Any]]
    ) -> None:
        """Build the training graph from streaming chunks and relationships."""
        logger.info("Building Sequential-ISNE graph from streaming data")
        
        # Add chunks and relationships to graph
        self.graph.add_chunks(chunks)
        self.graph.add_relationships(relationships)
        
        # Create node index mapping (solving the chunk-to-node mapping problem)
        content_chunks = [c for c in chunks if c.metadata.chunk_type == "content"]
        self.node_to_index = {chunk.chunk_id: i for i, chunk in enumerate(content_chunks)}
        self.index_to_node = {i: chunk.chunk_id for i, chunk in enumerate(content_chunks)}
        
        logger.info(f"Created consistent mapping for {len(self.node_to_index)} chunks")
        
        # Log graph statistics
        stats = self.graph.get_graph_statistics()
        logger.info(f"Graph statistics: {stats}")

    def build_graph_from_directory_graph(
        self,
        directory_graph: 'DirectoryGraph',
        chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Build Sequential-ISNE training graph from DirectoryGraph structure.
        
        This method integrates the directory-informed graph structure with 
        Sequential-ISNE training, providing proper graph-based ISNE instead 
        of sequential processing.
        
        Args:
            directory_graph: DirectoryGraph instance with bootstrapped structure
            chunks: List of chunk dictionaries with embeddings and metadata
        """
        logger.info("Building Sequential-ISNE graph from DirectoryGraph structure")
        
        # Clear existing graph
        self.graph = SequentialGraph()
        
        # Create chunk_id to file_path mapping
        chunk_to_file = {}
        file_to_chunks = defaultdict(list)
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            file_path = chunk['document_metadata']['file_path']
            chunk_to_file[chunk_id] = file_path
            file_to_chunks[file_path].append(chunk)
        
        # Convert directory graph nodes/edges to chunk relationships
        relationships = []
        chunk_counter = 0
        
        for file_path, file_chunks in file_to_chunks.items():
            # Add chunks for this file
            for chunk in file_chunks:
                chunk_counter += 1
        
        # Create relationships based on DirectoryGraph edges
        for node_a, node_b, edge_data in directory_graph.graph.edges(data=True):
            file_a = directory_graph.node_to_file.get(node_a)
            file_b = directory_graph.node_to_file.get(node_b)
            
            if file_a and file_b and file_a in file_to_chunks and file_b in file_to_chunks:
                # Create chunk-to-chunk relationships based on file relationships
                chunks_a = file_to_chunks[file_a]
                chunks_b = file_to_chunks[file_b]
                
                # Connect representative chunks (first chunk of each file)
                if chunks_a and chunks_b:
                    chunk_a = chunks_a[0]
                    chunk_b = chunks_b[0]
                    
                    relationship = {
                        'from_chunk_id': chunk_a['chunk_id'],
                        'to_chunk_id': chunk_b['chunk_id'],
                        'relationship_type': edge_data.get('edge_type', 'directory_informed'),
                        'confidence': edge_data.get('weight', 0.5),
                        'context': f"Directory connection: {edge_data.get('source', 'unknown')}"
                    }
                    relationships.append(relationship)
        
        # Convert chunks to StreamingChunk format for compatibility
        from src.streaming_processor import StreamingChunk, ChunkMetadata
        streaming_chunks = []
        
        for chunk in chunks:
            doc_meta = chunk.get('document_metadata', {})
            source_file = doc_meta.get('file_path', 'unknown')
            
            metadata = ChunkMetadata(
                chunk_id=chunk['chunk_id'],
                chunk_type='content',
                doc_path=source_file,
                directory=str(Path(source_file).parent),
                processing_order=chunk['chunk_id'],
                file_extension=Path(source_file).suffix
            )
            
            streaming_chunk = StreamingChunk(
                chunk_id=chunk['chunk_id'],
                content=chunk['content'],
                metadata=metadata,
                semantic_embedding=chunk.get('embedding', [])
            )
            streaming_chunks.append(streaming_chunk)
        
        # Build graph using existing method
        self.build_graph_from_chunks(streaming_chunks, relationships)
        
        logger.info(f"Built graph from DirectoryGraph: {len(streaming_chunks)} chunks, {len(relationships)} relationships")
        logger.info(f"Directory-informed relationship types: {list(set(r['relationship_type'] for r in relationships))}")
    
    def train_embeddings(self) -> Dict[str, Any]:
        """
        Train ISNE embeddings on the sequential graph.
        
        Returns training metrics and statistics.
        """
        if not HAS_TORCH:
            return self._train_embeddings_fallback()
        
        logger.info("Starting Sequential-ISNE training")
        
        # Initialize model with dynamic node count for academic scale
        device = self._get_device()
        num_nodes = len(self.node_to_index)
        logger.info(f"Initializing model for {num_nodes} nodes")
        self.model = SimpleISNEModel(self.config, num_nodes).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Prepare training data
        training_pairs = self._create_training_pairs()
        logger.info(f"Created {len(training_pairs)} training pairs")
        
        # Training loop
        losses = []
        for epoch in range(self.config.epochs):
            epoch_loss = self._train_epoch(training_pairs, optimizer, device)
            losses.append(epoch_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{self.config.epochs}, Loss: {epoch_loss:.4f}")
        
        # Extract trained embeddings
        self._extract_embeddings(device)
        
        training_metrics = {
            "final_loss": losses[-1] if losses else 0.0,
            "training_pairs": len(training_pairs),
            "epochs_completed": len(losses),
            "embedding_dimension": self.config.embedding_dim,
            "nodes_trained": len(self.node_to_index)
        }
        
        logger.info(f"Training completed: {training_metrics}")
        return training_metrics
    
    def _train_embeddings_fallback(self) -> Dict[str, Any]:
        """Fallback training without PyTorch (uses random embeddings)."""
        logger.warning("PyTorch not available, using random embeddings for validation")
        
        num_nodes = len(self.node_to_index)
        # Create consistent random embeddings based on node IDs
        np.random.seed(42)
        self.trained_embeddings = np.random.randn(num_nodes, self.config.embedding_dim)
        
        # Normalize embeddings
        norms = np.linalg.norm(self.trained_embeddings, axis=1, keepdims=True)
        self.trained_embeddings = self.trained_embeddings / norms
        
        return {
            "final_loss": 0.0,
            "training_pairs": 0,
            "epochs_completed": 0,
            "embedding_dimension": self.config.embedding_dim,
            "nodes_trained": num_nodes,
            "note": "Random embeddings (PyTorch not available)"
        }
    
    def _get_device(self) -> str:
        """Get the appropriate device for training."""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def _create_training_pairs(self) -> List[Tuple[int, int, float]]:
        """Create positive training pairs from graph edges."""
        pairs = []
        
        for from_id, to_id, edge_data in self.graph.graph.edges(data=True):
            if from_id in self.node_to_index and to_id in self.node_to_index:
                from_idx = self.node_to_index[from_id]
                to_idx = self.node_to_index[to_id]
                confidence = edge_data.get('confidence', 0.5)
                
                pairs.append((from_idx, to_idx, confidence))
        
        return pairs
    
    def _train_epoch(self, training_pairs: List[Tuple[int, int, float]], optimizer, device) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        # Simple batch training
        for i in range(0, len(training_pairs), self.config.batch_size):
            batch = training_pairs[i:i + self.config.batch_size]
            
            optimizer.zero_grad()
            loss = self._compute_batch_loss(batch, device)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(training_pairs)
    
    def _compute_batch_loss(self, batch: List[Tuple[int, int, float]], device) -> torch.Tensor:
        """Compute loss for a batch of training pairs."""
        batch_loss = 0.0
        
        for from_idx, to_idx, confidence in batch:
            # Get embeddings
            from_tensor = torch.tensor([from_idx], device=device)
            to_tensor = torch.tensor([to_idx], device=device)
            
            from_emb = self.model(from_tensor)
            to_emb = self.model(to_tensor)
            
            # Cosine similarity loss (simplified)
            similarity = torch.cosine_similarity(from_emb, to_emb)
            target_similarity = torch.tensor([confidence], device=device)
            
            loss = nn.MSELoss()(similarity, target_similarity)
            batch_loss += loss
        
        return batch_loss / len(batch)
    
    def _extract_embeddings(self, device) -> None:
        """Extract trained embeddings from the model."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(len(self.node_to_index)):
                node_tensor = torch.tensor([i], device=device)
                embedding = self.model(node_tensor).cpu().numpy()[0]
                embeddings.append(embedding)
        
        self.trained_embeddings = np.array(embeddings)
        logger.info(f"Extracted embeddings shape: {self.trained_embeddings.shape}")
    
    def get_chunk_embedding(self, chunk_id: int) -> Optional[np.ndarray]:
        """Get the ISNE embedding for a specific chunk."""
        if self.trained_embeddings is None:
            logger.warning("No trained embeddings available")
            return None
        
        if chunk_id not in self.node_to_index:
            logger.warning(f"Chunk {chunk_id} not found in training data")
            return None
        
        index = self.node_to_index[chunk_id]
        return self.trained_embeddings[index]
    
    def find_similar_chunks(
        self, 
        chunk_id: int, 
        k: int = 10, 
        similarity_threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Find chunks with similar ISNE embeddings."""
        if self.trained_embeddings is None:
            return []
        
        query_embedding = self.get_chunk_embedding(chunk_id)
        if query_embedding is None:
            return []
        
        # Compute similarities with all other chunks
        similarities = []
        for other_chunk_id in self.node_to_index.keys():
            if other_chunk_id != chunk_id:
                other_embedding = self.get_chunk_embedding(other_chunk_id)
                if other_embedding is not None:
                    similarity = np.dot(query_embedding, other_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(other_embedding)
                    )
                    if similarity >= similarity_threshold:
                        similarities.append((other_chunk_id, float(similarity)))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def save_model(self, output_path: str) -> None:
        """Save the trained model and embeddings."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_data = {
            'config': self.config.__dict__,
            'node_to_index': self.node_to_index,
            'index_to_node': self.index_to_node,
            'trained_embeddings': self.trained_embeddings.tolist() if self.trained_embeddings is not None else None,
            'graph_statistics': self.graph.get_graph_statistics()
        }
        
        with open(output_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Save graph separately
        graph_path = output_file.with_suffix('.graph')
        self.graph.save_graph(str(graph_path))
        
        logger.info(f"Saved Sequential-ISNE model to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo Sequential-ISNE training
    from streaming_processor import StreamingChunkProcessor
    from embeddings import EmbeddingManager, MockEmbeddingProvider
    import tempfile
    
    # Create sample test files
    test_files = {
        "src/auth/handler.py": "def authenticate_user():\n    return validate_token()",
        "src/auth/README.md": "# Authentication\nHandles user authentication with JWT tokens",
        "src/auth/tests.py": "def test_authenticate():\n    assert authenticate_user()",
        "src/utils/helpers.py": "def helper_function():\n    return True",
        "docs/guide.md": "# User Guide\nHow to authenticate users in the system"
    }
    
    print("=== Sequential-ISNE Demo ===")
    
    # Create temporary directory and files
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for rel_path, content in test_files.items():
            full_path = Path(temp_dir) / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            file_paths.append(str(full_path))
        
        # Process with StreamingChunkProcessor
        processor = StreamingChunkProcessor()
        chunks = list(processor.process_documents(file_paths))
        relationships = processor.get_sequential_relationships()
        
        print(f"Generated {len(chunks)} chunks and {len(relationships)} relationships")
        
        # Add semantic embeddings
        embedding_manager = EmbeddingManager(MockEmbeddingProvider())
        embedding_manager.embed_chunk_contents(chunks)
        
        # Train Sequential-ISNE
        config = TrainingConfig(epochs=20, embedding_dim=128)
        isne = SequentialISNE(config)
        isne.build_graph_from_chunks(chunks, relationships)
        
        training_metrics = isne.train_embeddings()
        print(f"Training metrics: {training_metrics}")
        
        # Test similarity search
        content_chunks = [c for c in chunks if c.metadata.chunk_type == "content"]
        if content_chunks:
            test_chunk_id = content_chunks[0].chunk_id
            similar_chunks = isne.find_similar_chunks(test_chunk_id, k=3)
            print(f"Similar chunks to {test_chunk_id}: {similar_chunks}")
        
        print("Sequential-ISNE demo completed successfully!")