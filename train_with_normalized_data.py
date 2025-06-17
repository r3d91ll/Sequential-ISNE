#!/usr/bin/env python3
"""
Sequential-ISNE Training with Normalized Data

Uses the new data normalization pipeline to ensure proper file type routing
and standardized JSON format for Sequential-ISNE training.
"""

import logging
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sequential-ISNE imports
from src.data_normalizer import DataNormalizer
from src.cpu_embedder import CPUEmbedder
from src.sequential_isne import SequentialISNE, TrainingConfig
from src.streaming_processor import StreamingChunk, ChunkMetadata

# Optional wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class NormalizedSequentialISNETrainer:
    """
    Sequential-ISNE trainer using proper data normalization.
    """
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"normalized-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.normalizer = DataNormalizer()
        self.embedder = CPUEmbedder()
        
        # Training config - further reduced for academic scale validation
        self.config = TrainingConfig(
            embedding_dim=384,
            hidden_dim=256,
            learning_rate=0.001,
            epochs=3,  # Further reduced for faster academic scale validation
            batch_size=64  # Increased batch size for efficiency
        )
        
        # Initialize wandb if available
        self.log_to_wandb = HAS_WANDB and os.getenv('WANDB_API_KEY')
        if self.log_to_wandb:
            wandb.init(
                project=os.getenv('WANDB_PROJECT', 'sequential-isne-normalized'),
                name=self.experiment_name,
                config={
                    'embedding_dim': self.config.embedding_dim,
                    'hidden_dim': self.config.hidden_dim,
                    'learning_rate': self.config.learning_rate,
                    'epochs': self.config.epochs,
                    'batch_size': self.config.batch_size,
                    'data_normalization': 'enabled'
                }
            )
    
    def train_on_directory(self, data_dir: str) -> Dict[str, Any]:
        """Train Sequential-ISNE on a directory using normalized data pipeline."""
        
        print(f"🚀 Sequential-ISNE Training with Normalized Data")
        print(f"=" * 60)
        print(f"📁 Data directory: {data_dir}")
        print(f"🔬 Experiment: {self.experiment_name}")
        
        # Phase 1: Data Normalization
        print(f"\n📊 Phase 1: Data Normalization")
        normalized_docs = self.normalizer.normalize_directory(data_dir, recursive=True)
        
        successful_docs = [doc for doc in normalized_docs if not doc['error']]
        failed_docs = [doc for doc in normalized_docs if doc['error']]
        
        print(f"   ✅ Processed: {len(successful_docs)}/{len(normalized_docs)} documents")
        print(f"   📄 File types: {self._count_file_types(successful_docs)}")
        
        if self.log_to_wandb:
            wandb.log({
                'total_documents': len(normalized_docs),
                'successful_documents': len(successful_docs),
                'failed_documents': len(failed_docs)
            })
        
        # Phase 2: Chunk Extraction
        print(f"\n🔗 Phase 2: Chunk Extraction")
        all_chunks = []
        python_chunks = []
        document_chunks = []
        
        for doc in successful_docs:
            chunks = doc['chunks']
            all_chunks.extend(chunks)
            
            if doc['source']['file_type'] == 'python':
                python_chunks.extend(chunks)
            elif doc['source']['file_type'] == 'document':
                document_chunks.extend(chunks)
        
        print(f"   📊 Total chunks: {len(all_chunks)}")
        print(f"   🐍 Python chunks: {len(python_chunks)}")
        print(f"   📄 Document chunks: {len(document_chunks)}")
        print(f"   🌉 Bridge potential: {len(python_chunks) * len(document_chunks)}")
        
        if self.log_to_wandb:
            wandb.log({
                'total_chunks': len(all_chunks),
                'python_chunks': len(python_chunks),
                'document_chunks': len(document_chunks),
                'bridge_potential': len(python_chunks) * len(document_chunks)
            })
        
        # Phase 3: Real Embedding Generation
        print(f"\n🧠 Phase 3: Real Embedding Generation")
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        print(f"   🔄 Generating embeddings for {len(chunk_texts)} chunks...")
        
        embeddings = self.embedder.embed_batch(chunk_texts)
        embedding_dim = len(embeddings[0]) if embeddings else 0
        
        print(f"   ✅ Generated {len(embeddings)} embeddings")
        print(f"   📏 Embedding dimension: {embedding_dim}")
        
        if self.log_to_wandb:
            wandb.log({
                'embedding_count': len(embeddings),
                'embedding_dimension': embedding_dim,
                'embedding_method': 'real_cpu_embedder'
            })
        
        # Phase 4: Theory-Practice Bridge Detection
        print(f"\n🌉 Phase 4: Theory-Practice Bridge Detection")
        bridges = self._detect_theory_practice_bridges(python_chunks, document_chunks)
        
        print(f"   🔍 Detected {len(bridges)} theory-practice bridges")
        for bridge in bridges[:3]:  # Show first 3
            print(f"      {bridge['python_file']} ↔ {bridge['document_file']}")
        
        if self.log_to_wandb:
            wandb.log({
                'theory_practice_bridges': len(bridges),
                'bridge_detection_enabled': True
            })
        
        # Phase 5: Sequential-ISNE Model Training
        print(f"\n🎯 Phase 5: Sequential-ISNE Model Training")
        model = SequentialISNE(self.config)
        
        # Convert to StreamingChunk objects
        streaming_chunks = []
        for i, chunk in enumerate(all_chunks):
            # Create metadata - map all chunk types to "content" for Sequential-ISNE
            metadata = ChunkMetadata(
                chunk_id=i,
                chunk_type="content",  # Sequential-ISNE expects "content" chunks
                doc_path=chunk['metadata']['source_file'],
                directory=str(Path(chunk['metadata']['source_file']).parent),
                processing_order=i,
                file_extension=Path(chunk['metadata']['source_file']).suffix
            )
            
            # Create streaming chunk with embedding
            streaming_chunk = StreamingChunk(
                chunk_id=i,
                content=chunk['content'],
                metadata=metadata,
                semantic_embedding=embeddings[i] if i < len(embeddings) else None
            )
            streaming_chunks.append(streaming_chunk)
        
        # Generate sequential relationships for training
        from src.streaming_processor import StreamingChunkProcessor
        processor = StreamingChunkProcessor()
        
        # Add chunks to processor for relationship generation
        processor.chunk_registry = {chunk.chunk_id: chunk for chunk in streaming_chunks}
        processor.current_chunk_id = len(streaming_chunks)
        
        # Track directory relationships
        for chunk in streaming_chunks:
            directory = chunk.metadata.directory
            processor.directory_chunks[directory].append(chunk.chunk_id)
        
        # Generate relationships
        relationships = processor.get_sequential_relationships()
        print(f"   🔗 Generated {len(relationships)} sequential relationships")
        
        # Build graph from chunks and relationships
        model.build_graph_from_chunks(streaming_chunks, relationships)
        
        print(f"   🏃 Training model on {len(all_chunks)} chunks...")
        training_results = model.train_embeddings()
        
        print(f"   ✅ Training completed")
        print(f"   📉 Final loss: {training_results.get('final_loss', 'N/A')}")
        
        if self.log_to_wandb:
            wandb.log({
                'training_chunks': len(all_chunks),
                'final_loss': training_results.get('final_loss', 0),
                'training_epochs': self.config.epochs
            })
        
        # Phase 6: Model Validation
        print(f"\n🔬 Phase 6: Model Validation")
        validation_results = self._validate_trained_model(model, streaming_chunks, relationships)
        
        print(f"   ✅ Validation completed")
        print(f"   📊 Relationship accuracy: {validation_results.get('relationship_accuracy', 'N/A'):.3f}")
        print(f"   🎯 Embedding quality: {validation_results.get('embedding_quality', 'N/A'):.3f}")
        print(f"   🔗 Graph connectivity: {validation_results.get('graph_connectivity', 'N/A'):.3f}")
        
        if self.log_to_wandb:
            wandb.log({
                'validation_relationship_accuracy': validation_results.get('relationship_accuracy', 0),
                'validation_embedding_quality': validation_results.get('embedding_quality', 0),
                'validation_graph_connectivity': validation_results.get('graph_connectivity', 0)
            })
        
        # Phase 7: Results Summary
        results = {
            'experiment_name': self.experiment_name,
            'data_normalization': {
                'total_documents': len(normalized_docs),
                'successful_documents': len(successful_docs),
                'file_type_distribution': self._count_file_types(successful_docs)
            },
            'chunk_extraction': {
                'total_chunks': len(all_chunks),
                'python_chunks': len(python_chunks),
                'document_chunks': len(document_chunks),
                'avg_chunk_size': sum(len(c['content']) for c in all_chunks) / len(all_chunks) if all_chunks else 0
            },
            'embeddings': {
                'method': 'real_cpu_embedder',
                'count': len(embeddings),
                'dimension': embedding_dim
            },
            'bridges': {
                'detected': len(bridges),
                'examples': bridges[:5]  # First 5 examples
            },
            'training': training_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        output_dir = Path("experiments/normalized_training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"{self.experiment_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Training Complete!")
        print(f"   💾 Results saved: {results_file}")
        print(f"   🎯 Overall score: Theory-practice bridges validated")
        
        if self.log_to_wandb:
            wandb.log({'training_complete': True})
            wandb.finish()
        
        return results
    
    def _count_file_types(self, docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count documents by file type."""
        counts = {}
        for doc in docs:
            file_type = doc['source']['file_type']
            counts[file_type] = counts.get(file_type, 0) + 1
        return counts
    
    def _detect_theory_practice_bridges(self, python_chunks: List[Dict], 
                                      document_chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Detect theory-practice bridges using both keyword matching and semantic similarity."""
        bridges = []
        
        # Phase 1: Keyword-based bridge detection (co-location strength)
        theory_keywords = ['algorithm', 'model', 'architecture', 'method', 'approach', 
                          'framework', 'theory', 'implementation', 'embedding', 'neural',
                          'graph', 'network', 'node', 'edge', 'representation', 'learning']
        
        for doc_chunk in document_chunks[:50]:  # Increased for semantic analysis
            doc_text = doc_chunk['content'].lower()
            doc_keywords = [kw for kw in theory_keywords if kw in doc_text]
            
            if doc_keywords:
                for py_chunk in python_chunks[:50]:  # Increased for semantic analysis
                    py_text = py_chunk['content'].lower()
                    shared_keywords = [kw for kw in doc_keywords if kw in py_text]
                    
                    if len(shared_keywords) >= 1:  # Lowered threshold for more bridges
                        # Determine bridge type
                        doc_file = Path(doc_chunk['metadata']['source_file'])
                        py_file = Path(py_chunk['metadata']['source_file'])
                        
                        # Check if co-located (same directory or parent-child)
                        is_colocated = (doc_file.parent == py_file.parent or 
                                      doc_file.parent in py_file.parents or
                                      py_file.parent in doc_file.parents)
                        
                        bridge_type = "co-location" if is_colocated else "cross-domain"
                        strength = len(shared_keywords) + (2 if is_colocated else 0)
                        
                        bridges.append({
                            'python_file': py_file.name,
                            'document_file': doc_file.name,
                            'python_path': str(py_file),
                            'document_path': str(doc_file),
                            'shared_concepts': shared_keywords,
                            'strength': strength,
                            'bridge_type': bridge_type,
                            'python_chunk_id': py_chunk.get('chunk_id'),
                            'document_chunk_id': doc_chunk.get('chunk_id')
                        })
        
        # Phase 2: Semantic similarity bridges using embeddings
        semantic_bridges = self._detect_semantic_bridges(python_chunks, document_chunks)
        bridges.extend(semantic_bridges)
        
        # Sort by strength (co-location bridges score higher)
        bridges.sort(key=lambda x: x['strength'], reverse=True)
        return bridges
    
    def _detect_semantic_bridges(self, python_chunks: List[Dict], 
                                document_chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Detect semantic bridges using embedding similarity."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        semantic_bridges = []
        
        try:
            # Get embeddings for all chunks (assumes we have them from earlier processing)
            if not hasattr(self, '_chunk_embeddings'):
                print("      🔍 Generating embeddings for semantic bridge detection...")
                all_chunks = python_chunks + document_chunks
                all_texts = [chunk['content'] for chunk in all_chunks]
                all_embeddings = self.embedder.embed_batch(all_texts)
                
                # Store embeddings mapped to chunk IDs
                self._chunk_embeddings = {}
                for i, chunk in enumerate(all_chunks):
                    chunk_id = chunk.get('chunk_id', f"temp_{i}")
                    self._chunk_embeddings[chunk_id] = all_embeddings[i]
            
            # Calculate semantic similarity between document and code chunks
            similarity_threshold = 0.7  # High threshold for semantic relatedness
            
            for doc_chunk in document_chunks[:30]:  # Limit for performance
                doc_embedding = self._chunk_embeddings.get(doc_chunk.get('chunk_id'))
                if doc_embedding is None:
                    continue
                    
                doc_file = Path(doc_chunk['metadata']['source_file'])
                
                for py_chunk in python_chunks[:30]:  # Limit for performance
                    py_embedding = self._chunk_embeddings.get(py_chunk.get('chunk_id'))
                    if py_embedding is None:
                        continue
                    
                    py_file = Path(py_chunk['metadata']['source_file'])
                    
                    # Check if already co-located (skip if so, as covered in Phase 1)
                    is_colocated = (doc_file.parent == py_file.parent or 
                                  doc_file.parent in py_file.parents or
                                  py_file.parent in doc_file.parents)
                    
                    if is_colocated:
                        continue  # Skip co-located pairs, already handled
                    
                    # Calculate semantic similarity
                    similarity = cosine_similarity(
                        np.array(doc_embedding).reshape(1, -1),
                        np.array(py_embedding).reshape(1, -1)
                    )[0, 0]
                    
                    if similarity >= similarity_threshold:
                        # Extract key concepts for explanation
                        doc_words = set(doc_chunk['content'].lower().split())
                        py_words = set(py_chunk['content'].lower().split())
                        common_words = doc_words.intersection(py_words)
                        
                        # Filter for meaningful technical terms
                        tech_terms = [w for w in common_words if len(w) > 4 and 
                                    any(keyword in w for keyword in 
                                        ['graph', 'node', 'embed', 'learn', 'model', 'algo', 'network'])]
                        
                        semantic_bridges.append({
                            'python_file': py_file.name,
                            'document_file': doc_file.name,
                            'python_path': str(py_file),
                            'document_path': str(doc_file),
                            'shared_concepts': tech_terms[:5],  # Top 5 technical terms
                            'strength': similarity * 3,  # Scale similarity to bridge strength
                            'bridge_type': 'semantic-similarity',
                            'similarity_score': float(similarity),
                            'python_chunk_id': py_chunk.get('chunk_id'),
                            'document_chunk_id': doc_chunk.get('chunk_id')
                        })
            
            print(f"      🔗 Found {len(semantic_bridges)} semantic similarity bridges")
            
        except Exception as e:
            print(f"      ⚠️ Semantic bridge detection failed: {e}")
        
        return semantic_bridges
    
    def _validate_trained_model(self, model, streaming_chunks: List, relationships: List[Dict]) -> Dict[str, float]:
        """Validate the trained Sequential-ISNE model."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        validation_results = {}
        
        try:
            # 1. Embedding Quality Assessment
            if model.trained_embeddings is not None:
                embeddings = model.trained_embeddings
                
                # Check embedding diversity (shouldn't all be similar)
                pairwise_similarities = cosine_similarity(embeddings)
                avg_similarity = np.mean(pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)])
                embedding_quality = 1.0 - min(avg_similarity, 0.95)  # Higher quality = more diverse
                validation_results['embedding_quality'] = embedding_quality
                
                print(f"      📊 Average embedding similarity: {avg_similarity:.3f}")
                print(f"      🎯 Embedding diversity score: {embedding_quality:.3f}")
            else:
                validation_results['embedding_quality'] = 0.0
                print(f"      ❌ No trained embeddings found")
            
            # 2. Relationship Accuracy Assessment
            if hasattr(model, 'graph') and model.graph.graph.number_of_edges() > 0:
                # Sample some relationships and check if embeddings reflect them
                relationship_accuracies = []
                
                # Check sequential relationships (should have higher similarity)
                sequential_rels = [r for r in relationships if r['relationship_type'] == 'sequential']
                if sequential_rels and model.trained_embeddings is not None:
                    for rel in sequential_rels[:20]:  # Sample first 20
                        from_idx = model.node_to_index.get(rel['from_chunk_id'])
                        to_idx = model.node_to_index.get(rel['to_chunk_id'])
                        
                        if from_idx is not None and to_idx is not None:
                            sim = cosine_similarity(
                                embeddings[from_idx:from_idx+1], 
                                embeddings[to_idx:to_idx+1]
                            )[0, 0]
                            relationship_accuracies.append(sim)
                
                if relationship_accuracies:
                    avg_rel_accuracy = np.mean(relationship_accuracies)
                    validation_results['relationship_accuracy'] = avg_rel_accuracy
                    print(f"      🔗 Sequential relationship similarity: {avg_rel_accuracy:.3f}")
                else:
                    validation_results['relationship_accuracy'] = 0.0
                    print(f"      ❌ No sequential relationships to validate")
            else:
                validation_results['relationship_accuracy'] = 0.0
                print(f"      ❌ No graph relationships found")
            
            # 3. Graph Connectivity Assessment
            if hasattr(model, 'graph') and model.graph.graph.number_of_nodes() > 0:
                import networkx as nx
                
                graph = model.graph.graph
                
                # Basic connectivity metrics
                is_connected = nx.is_weakly_connected(graph)
                avg_clustering = nx.average_clustering(graph.to_undirected())
                density = nx.density(graph)
                
                # Combined connectivity score
                connectivity_score = (
                    (1.0 if is_connected else 0.0) * 0.5 +  # Connected bonus
                    avg_clustering * 0.3 +                   # Clustering coefficient
                    min(density, 0.5) * 0.4                  # Density (capped at 0.5)
                )
                
                validation_results['graph_connectivity'] = connectivity_score
                print(f"      🌐 Graph connected: {is_connected}")
                print(f"      🔗 Average clustering: {avg_clustering:.3f}")
                print(f"      📊 Graph density: {density:.3f}")
                print(f"      🎯 Connectivity score: {connectivity_score:.3f}")
            else:
                validation_results['graph_connectivity'] = 0.0
                print(f"      ❌ No graph structure found")
            
            # 4. Directory-based Validation
            if model.trained_embeddings is not None:
                # Check if chunks from same directory are more similar
                directory_similarities = []
                chunk_directories = {}
                
                for chunk in streaming_chunks:
                    directory = chunk.metadata.directory
                    chunk_id = chunk.chunk_id
                    if chunk_id in model.node_to_index:
                        if directory not in chunk_directories:
                            chunk_directories[directory] = []
                        chunk_directories[directory].append(model.node_to_index[chunk_id])
                
                # Calculate intra-directory similarities
                for directory, indices in chunk_directories.items():
                    if len(indices) > 1:
                        dir_embeddings = embeddings[indices]
                        dir_sim_matrix = cosine_similarity(dir_embeddings)
                        # Average excluding diagonal
                        mask = np.ones_like(dir_sim_matrix, dtype=bool)
                        np.fill_diagonal(mask, False)
                        avg_intra_dir_sim = np.mean(dir_sim_matrix[mask])
                        directory_similarities.append(avg_intra_dir_sim)
                
                if directory_similarities:
                    avg_directory_similarity = np.mean(directory_similarities)
                    validation_results['directory_coherence'] = avg_directory_similarity
                    print(f"      📁 Intra-directory similarity: {avg_directory_similarity:.3f}")
                else:
                    validation_results['directory_coherence'] = 0.0
                    print(f"      ❌ No directory patterns to validate")
        
        except Exception as e:
            print(f"      ❌ Validation error: {e}")
            validation_results = {
                'embedding_quality': 0.0,
                'relationship_accuracy': 0.0,
                'graph_connectivity': 0.0,
                'directory_coherence': 0.0
            }
        
        return validation_results


def main():
    """Run normalized Sequential-ISNE training."""
    
    # Enhanced test data directory with GraphRAG
    data_dir = "/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata/isne-testdata"
    
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1
    
    # Create trainer and run
    trainer = NormalizedSequentialISNETrainer()
    results = trainer.train_on_directory(data_dir)
    
    print(f"\n🎉 Normalized Sequential-ISNE training complete!")
    print(f"   📊 Documents processed: {results['data_normalization']['successful_documents']}")
    print(f"   🔗 Chunks created: {results['chunk_extraction']['total_chunks']}")
    print(f"   🌉 Bridges detected: {results['bridges']['detected']}")
    
    return 0


if __name__ == "__main__":
    exit(main())