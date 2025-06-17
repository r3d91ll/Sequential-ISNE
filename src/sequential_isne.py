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
from typing import List, Dict, Any, Tuple, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from collections import defaultdict

# Type checking imports
if TYPE_CHECKING:
    from src.streaming_processor import StreamingChunk
    from directory_graph import DirectoryGraph

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
            logger.warning("PyTorch not available, using fallback training")
            return self._train_embeddings_fallback()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache before training")
        
        logger.info("Starting Sequential-ISNE training with PyTorch")
        logger.info("🔥 DEBUG: Sequential-ISNE training started with PyTorch")
        
        # Initialize model with dynamic node count for academic scale
        device = self._get_device()
        num_nodes = len(self.node_to_index)
        logger.info(f"Initializing model for {num_nodes} nodes on device: {device}")
        
        try:
            self.model = SimpleISNEModel(self.config, num_nodes).to(device)
            logger.info(f"Model initialized successfully: {self.model}")
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            logger.info(f"Optimizer initialized with LR: {self.config.learning_rate}")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return self._train_embeddings_fallback()
        
        # Prepare training data
        logger.info("Creating training pairs from graph edges...")
        logger.info(f"🔍 DEBUG: Creating training pairs from {self.graph.graph.number_of_edges()} graph edges")
        training_pairs = self._create_training_pairs()
        logger.info(f"Created {len(training_pairs)} training pairs")
        logger.info(f"📊 DEBUG: Created {len(training_pairs)} training pairs")
        
        if len(training_pairs) == 0:
            logger.warning("No training pairs found! Graph may have no edges or mapping issues.")
            logger.warning(f"❌ DEBUG: NO TRAINING PAIRS! Graph edges: {self.graph.graph.number_of_edges()}, Node mappings: {len(self.node_to_index)}")
            logger.info(f"Graph edges: {self.graph.graph.number_of_edges()}")
            logger.info(f"Node mapping size: {len(self.node_to_index)}")
            return self._train_embeddings_fallback()
        
        # Training loop
        logger.info(f"Starting training loop for {self.config.epochs} epochs...")
        logger.info(f"🔄 DEBUG: Starting training loop for {self.config.epochs} epochs")
        losses = []
        
        try:
            for epoch in range(self.config.epochs):
                logger.debug(f"Starting epoch {epoch}")
                epoch_loss = self._train_epoch(training_pairs, optimizer, device)
                losses.append(epoch_loss)
                
                if epoch % 10 == 0:  # More frequent updates to see loss progression
                    logger.info(f"Epoch {epoch}/{self.config.epochs}, Loss: {epoch_loss:.4f}")
                    logger.info(f"📈 DEBUG: Epoch {epoch}/{self.config.epochs}, Loss: {epoch_loss:.4f}")
                    
                    # Sample embedding similarity check
                    if epoch > 0 and epoch % 20 == 0:
                        self.model.eval()
                        with torch.no_grad():
                            sample_emb1 = self.model(torch.tensor([0], device=device))
                            sample_emb2 = self.model(torch.tensor([1], device=device))
                            sample_emb3 = self.model(torch.tensor([2], device=device))
                            
                            sim_12 = torch.cosine_similarity(sample_emb1, sample_emb2).item()
                            sim_13 = torch.cosine_similarity(sample_emb1, sample_emb3).item()
                            sim_23 = torch.cosine_similarity(sample_emb2, sample_emb3).item()
                            
                            logger.info(f"🔍 DEBUG: Sample similarities - 0↔1: {sim_12:.4f}, 0↔2: {sim_13:.4f}, 1↔2: {sim_23:.4f}")
                        self.model.train()
                
            logger.info(f"Training completed successfully! Total epochs: {len(losses)}")
            logger.info(f"✅ DEBUG: Training completed! Total epochs: {len(losses)}")
        except Exception as e:
            logger.error(f"Training failed during epoch {len(losses)}: {e}")
            logger.error(f"❌ DEBUG: Training failed during epoch {len(losses)}: {e}")
            logger.info("Falling back to random embeddings")
            return self._train_embeddings_fallback()
        
        # Extract trained embeddings
        logger.info("Extracting trained embeddings from model...")
        try:
            self._extract_embeddings(device)
            logger.info(f"Successfully extracted embeddings: shape {self.trained_embeddings.shape if self.trained_embeddings is not None else 'None'}")
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return self._train_embeddings_fallback()
        
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
        logger.warning("=== USING FALLBACK RANDOM EMBEDDINGS ===")
        logger.warning("🚨 DEBUG: === USING FALLBACK RANDOM EMBEDDINGS ===")
        logger.warning("This means PyTorch training failed or wasn't available!")
        logger.warning("🚨 DEBUG: This means PyTorch training failed or wasn't available!")
        
        num_nodes = len(self.node_to_index)
        logger.info(f"Creating random embeddings for {num_nodes} nodes, dim={self.config.embedding_dim}")
        
        # Create consistent random embeddings based on node IDs
        np.random.seed(42)
        self.trained_embeddings = np.random.randn(num_nodes, self.config.embedding_dim)
        logger.info(f"Generated random embeddings shape: {self.trained_embeddings.shape}")
        
        # Normalize embeddings
        norms = np.linalg.norm(self.trained_embeddings, axis=1, keepdims=True)
        logger.info(f"Embedding norms before normalization: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        self.trained_embeddings = self.trained_embeddings / norms
        
        # Verify normalization
        new_norms = np.linalg.norm(self.trained_embeddings, axis=1)
        logger.info(f"Embedding norms after normalization: min={new_norms.min():.4f}, max={new_norms.max():.4f}, mean={new_norms.mean():.4f}")
        
        # Check similarity distribution
        sample_similarities = []
        for i in range(min(10, num_nodes)):
            for j in range(i+1, min(10, num_nodes)):
                sim = np.dot(self.trained_embeddings[i], self.trained_embeddings[j])
                sample_similarities.append(sim)
        
        if sample_similarities:
            logger.info(f"Sample cosine similarities: min={min(sample_similarities):.4f}, max={max(sample_similarities):.4f}, mean={np.mean(sample_similarities):.4f}")
        
        logger.warning("=== END FALLBACK EMBEDDING GENERATION ===")
        
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
        total_edges = 0
        valid_edges = 0
        
        logger.info(f"Processing {self.graph.graph.number_of_edges()} graph edges for training pairs")
        logger.info(f"Available chunk mappings: {len(self.node_to_index)}")
        
        # Debug: Show first few node mappings
        sample_nodes = list(self.node_to_index.keys())[:5]
        logger.info(f"Sample node IDs in mapping: {sample_nodes}")
        
        # Debug: Show first few graph edges
        sample_edges = list(self.graph.graph.edges(data=True))[:5]
        logger.info(f"Sample graph edges: {[(f, t, d.get('relationship_type', 'unknown')) for f, t, d in sample_edges]}")
        
        for from_id, to_id, edge_data in self.graph.graph.edges(data=True):
            total_edges += 1
            
            if from_id in self.node_to_index and to_id in self.node_to_index:
                from_idx = self.node_to_index[from_id]
                to_idx = self.node_to_index[to_id]
                confidence = edge_data.get('confidence', 0.5)
                
                pairs.append((from_idx, to_idx, confidence))
                valid_edges += 1
            else:
                if total_edges <= 3:  # Log first few missing mappings
                    logger.warning(f"Missing mapping for edge: {from_id} -> {to_id}")
                    logger.warning(f"  from_id in mapping: {from_id in self.node_to_index}")
                    logger.warning(f"  to_id in mapping: {to_id in self.node_to_index}")
        
        logger.info(f"Training pairs created: {len(pairs)} valid / {total_edges} total edges")
        logger.info(f"Mapping success rate: {valid_edges/total_edges*100:.1f}%")
        
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
        """Compute ISNE loss with skip-gram objective and negative sampling."""
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Get all node indices for negative sampling
        all_indices = list(range(len(self.node_to_index)))
        
        for from_idx, to_idx, confidence in batch:
            # Get embeddings for positive pair
            from_tensor = torch.tensor([from_idx], device=device)
            to_tensor = torch.tensor([to_idx], device=device)
            
            from_emb = self.model(from_tensor)  # Shape: [1, embedding_dim]
            to_emb = self.model(to_tensor)      # Shape: [1, embedding_dim]
            
            # Positive loss: -log(σ(u_i^T * v_j))
            positive_score = torch.sum(from_emb * to_emb, dim=1)  # Dot product
            positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-8)  # Add epsilon for numerical stability
            
            # Negative sampling: sample k negative nodes
            negative_loss = torch.tensor(0.0, device=device)
            num_negative_samples = min(self.config.negative_samples, len(all_indices) - 2)  # Exclude from_idx and to_idx
            
            if num_negative_samples > 0:
                # Sample negative nodes (not from_idx or to_idx)
                available_negatives = [idx for idx in all_indices if idx != from_idx and idx != to_idx]
                if len(available_negatives) >= num_negative_samples:
                    negative_indices = torch.tensor(
                        torch.randperm(len(available_negatives))[:num_negative_samples].tolist(),
                        device=device
                    )
                    negative_indices = torch.tensor([available_negatives[i] for i in negative_indices], device=device)
                    
                    # Get negative embeddings
                    negative_embs = self.model(negative_indices)  # Shape: [num_negative_samples, embedding_dim]
                    
                    # Negative loss: -Σ log(σ(-u_i^T * v_k))
                    negative_scores = torch.sum(from_emb * negative_embs, dim=1)  # Shape: [num_negative_samples]
                    negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_scores) + 1e-8))
            
            # Combine positive and negative losses (weighted by confidence)
            sample_loss = confidence * positive_loss + negative_loss
            total_loss = total_loss + sample_loss
        
        return total_loss / len(batch)
    
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
    
    def compute_all_similarities_gpu(self, similarity_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        GPU-accelerated computation of all pairwise similarities above threshold.
        
        Returns:
            List of (chunk_id_a, chunk_id_b, similarity) tuples
        """
        if self.trained_embeddings is None:
            logger.warning("No trained embeddings available for GPU similarity computation")
            return []
        
        if not HAS_TORCH:
            logger.warning("PyTorch not available, falling back to CPU computation")
            return self._compute_all_similarities_cpu(similarity_threshold)
        
        device = self._get_device()
        logger.info(f"Computing all pairwise similarities on {device}")
        
        # Convert embeddings to GPU tensor
        embeddings_tensor = torch.tensor(self.trained_embeddings, device=device, dtype=torch.float32)
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
        
        # Compute similarity matrix in batches to manage GPU memory
        # Adjust batch size based on available GPU memory
        if device == "cuda":
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                # Estimate batch size based on GPU memory (conservative)
                max_batch_size = min(200, int(gpu_memory_gb * 10))
                logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB, using batch size: {max_batch_size}")
            except:
                max_batch_size = 100
        else:
            max_batch_size = 50
            
        batch_size = min(max_batch_size, embeddings_normalized.size(0))
        chunk_ids = list(self.node_to_index.keys())
        similar_pairs = []
        
        logger.info(f"Processing {len(chunk_ids)} chunks in batches of {batch_size} on {device}")
        
        for i in range(0, len(chunk_ids), batch_size):
            batch_end = min(i + batch_size, len(chunk_ids))
            batch_embeddings = embeddings_normalized[i:batch_end]
            
            # Compute similarities between this batch and all embeddings
            similarities = torch.mm(batch_embeddings, embeddings_normalized.t())
            
            # Find pairs above threshold (excluding self-similarities)
            for batch_idx, global_idx_a in enumerate(range(i, batch_end)):
                chunk_id_a = chunk_ids[global_idx_a]
                
                # Only look at upper triangle to avoid duplicates
                for global_idx_b in range(global_idx_a + 1, len(chunk_ids)):
                    chunk_id_b = chunk_ids[global_idx_b]
                    similarity = similarities[batch_idx, global_idx_b].item()
                    
                    if similarity >= similarity_threshold:
                        similar_pairs.append((chunk_id_a, chunk_id_b, similarity))
            
            if i % (batch_size * 5) == 0:
                logger.info(f"GPU similarity progress: {batch_end}/{len(chunk_ids)} chunks processed")
        
        logger.info(f"GPU computation found {len(similar_pairs)} similar pairs above threshold {similarity_threshold}")
        
        # Clear GPU cache after computation
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache after similarity computation")
        
        return similar_pairs
    
    def _compute_all_similarities_cpu(self, similarity_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """Fallback CPU computation of all pairwise similarities."""
        logger.info("Computing all pairwise similarities on CPU (fallback)")
        
        chunk_ids = list(self.node_to_index.keys())
        similar_pairs = []
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.trained_embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.trained_embeddings / norms
        
        for i, chunk_id_a in enumerate(chunk_ids):
            if i % 100 == 0:
                logger.info(f"CPU similarity progress: {i}/{len(chunk_ids)} chunks processed")
            
            for j in range(i + 1, len(chunk_ids)):
                chunk_id_b = chunk_ids[j]
                
                # Cosine similarity using normalized embeddings
                similarity = np.dot(normalized_embeddings[i], normalized_embeddings[j])
                
                if similarity >= similarity_threshold:
                    similar_pairs.append((chunk_id_a, chunk_id_b, similarity))
        
        logger.info(f"CPU computation found {len(similar_pairs)} similar pairs above threshold {similarity_threshold}")
        return similar_pairs
    
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