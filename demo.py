#!/usr/bin/env python3
"""
Sequential-ISNE Complete Pipeline Demo

Demonstrates the complete Sequential-ISNE pipeline:
1. Data normalization with Docling
2. Unified chunking (Chonky + AST)
3. CodeBERT embedding for both text and code
4. Sequential-ISNE training
5. Theory-practice bridge detection

Usage:
    python demo.py [data_directory]
    
Example:
    python demo.py /path/to/dataset
"""

import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Import our simplified components
from src.data_normalizer import DataNormalizer
from chunker import UnifiedChunker
from embedder import CodeBERTEmbedder
from src.sequential_isne import SequentialISNE, TrainingConfig
from src.streaming_processor import StreamingChunk, ChunkMetadata, StreamingChunkProcessor


class SequentialISNEDemo:
    """
    Complete Sequential-ISNE demonstration pipeline.
    Shows data normalization contribution and ISNE training.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        self.experiment_name = f"sequential-isne-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.normalizer = DataNormalizer()
        self.chunker = UnifiedChunker(self.config['chunking'])
        self.embedder = CodeBERTEmbedder(self.config['embedding'])
        
        # Training configuration
        self.training_config = TrainingConfig(
            embedding_dim=self.config['model']['embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            learning_rate=self.config['model']['learning_rate'],
            epochs=self.config['model']['epochs'],
            batch_size=self.config['model']['batch_size'],
            negative_samples=self.config['model']['negative_samples'],
            device=self.config['model']['device']
        )
        
        self.logger.info(f"Sequential-ISNE Demo initialized: {self.experiment_name}")
        self.logger.info(f"Configuration loaded from: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        return {
            'chunking': {'text_chunk_size': 512, 'text_overlap': 50},
            'embedding': {'model_name': 'microsoft/codebert-base', 'max_length': 512, 'normalize': True},
            'model': {
                'embedding_dim': 768, 'hidden_dim': 256, 'learning_rate': 0.001,
                'epochs': 5, 'batch_size': 16, 'negative_samples': 5, 'device': 'auto'
            },
            'bridges': {
                'theory_keywords': ['algorithm', 'model', 'graph', 'network', 'embedding'],
                'similarity_threshold': 0.7, 'max_sample_chunks': 50
            },
            'output': {'save_results': True, 'results_directory': 'experiments'}
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Save logs if configured
        if self.config.get('logging', {}).get('save_logs', False):
            log_dir = Path(self.config.get('logging', {}).get('log_directory', 'logs'))
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"{self.experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
    
    def run_complete_pipeline(self, data_directory: str) -> Dict[str, Any]:
        """
        Run the complete Sequential-ISNE pipeline on a dataset.
        
        Args:
            data_directory: Path to directory containing PDFs, code, and documents
            
        Returns:
            Complete results dictionary with all metrics and outputs
        """
        data_path = Path(data_directory)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_directory}")
        
        print("=" * 80)
        print("🚀 SEQUENTIAL-ISNE COMPLETE PIPELINE DEMONSTRATION")
        print("=" * 80)
        print(f"📁 Dataset: {data_directory}")
        print(f"🔬 Experiment: {self.experiment_name}")
        print(f"⏰ Started: {datetime.now().isoformat()}")
        print()
        
        # Phase 1: Data Normalization with Docling
        print("📊 PHASE 1: Data Normalization (Docling)")
        print("-" * 50)
        normalized_docs = self._normalize_data(data_directory)
        
        # Phase 2: Unified Chunking (Chonky + AST)
        print("\n🔗 PHASE 2: Unified Chunking (Chonky + AST)")
        print("-" * 50)
        all_chunks = self._chunk_documents(normalized_docs)
        
        # Phase 3: CodeBERT Embedding
        print("\n🧠 PHASE 3: CodeBERT Embedding (Text + Code)")
        print("-" * 50)
        embedded_chunks = self._embed_chunks(all_chunks)
        
        # Phase 4: Theory-Practice Bridge Detection
        print("\n🌉 PHASE 4: Theory-Practice Bridge Detection")
        print("-" * 50)
        bridges = self._detect_bridges(embedded_chunks)
        
        # Phase 5: Sequential-ISNE Training
        print("\n🎯 PHASE 5: Sequential-ISNE Training")
        print("-" * 50)
        model, training_results = self._train_model(embedded_chunks)
        
        # Phase 6: Validation and Analysis
        print("\n🔬 PHASE 6: Model Validation")
        print("-" * 50)
        validation_results = self._validate_model(model, embedded_chunks, bridges)
        
        # Phase 7: Results Compilation
        print("\n📊 PHASE 7: Results Summary")
        print("-" * 50)
        results = self._compile_results(
            normalized_docs, all_chunks, embedded_chunks, 
            bridges, training_results, validation_results
        )
        
        # Save results
        if self.config.get('output', {}).get('save_results', True):
            self._save_results(results)
        
        print("\n🎉 SEQUENTIAL-ISNE PIPELINE COMPLETE!")
        self._display_summary(results)
        
        return results
    
    def _normalize_data(self, data_directory: str) -> List[Dict[str, Any]]:
        """Phase 1: Normalize documents using Docling."""
        print("   🔄 Processing documents with Docling...")
        
        normalized_docs = self.normalizer.normalize_directory(
            data_directory, 
            recursive=self.config.get('normalization', {}).get('recursive', True)
        )
        
        successful_docs = [doc for doc in normalized_docs if not doc['error']]
        failed_docs = [doc for doc in normalized_docs if doc['error']]
        
        print(f"   ✅ Processed: {len(successful_docs)}/{len(normalized_docs)} documents")
        print(f"   📄 File types: {self._count_file_types(successful_docs)}")
        
        if failed_docs:
            print(f"   ⚠️  Failed: {len(failed_docs)} documents")
        
        return successful_docs
    
    def _chunk_documents(self, normalized_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 2: Chunk documents using unified chunker."""
        print("   🔄 Chunking with Chonky (text) and AST (Python)...")
        
        all_chunks = []
        python_chunks = 0
        text_chunks = 0
        
        for doc in normalized_docs:
            file_type = doc['source']['file_type']
            # Use raw content for chunking, processed content is already parsed
            content = doc['content']['raw_content'] or doc['content']['processed_content']
            source_file = doc['source']['file_path']
            
            # Skip if no content or error
            if not content or doc.get('error'):
                continue
            
            # For text content, ensure it's a string
            if isinstance(content, dict):
                # This is parsed content, skip for now or convert
                continue
            
            # Chunk based on type
            chunks = self.chunker.chunk_content(content, file_type, source_file)
            
            # Add document metadata to chunks
            for chunk in chunks:
                chunk['document_metadata'] = doc['source']
                chunk['chunk_id'] = len(all_chunks)  # Global sequential ID
                all_chunks.append(chunk)
                
                if file_type == 'python':
                    python_chunks += 1
                else:
                    text_chunks += 1
        
        print(f"   📊 Total chunks: {len(all_chunks)}")
        print(f"   🐍 Python chunks: {python_chunks}")
        print(f"   📄 Text chunks: {text_chunks}")
        print(f"   🌉 Bridge potential: {python_chunks * text_chunks}")
        
        # Show chunking statistics
        stats = self.chunker.get_chunking_stats(all_chunks)
        print(f"   📏 Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
        print(f"   🔧 Chunking methods: {stats['chunking_methods']}")
        
        return all_chunks
    
    def _embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 3: Generate CodeBERT embeddings."""
        print("   🔄 Generating CodeBERT embeddings...")
        
        # Extract texts for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings using CodeBERT
        embeddings = self.embedder.embed_batch(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding
            chunk_copy['embedding_info'] = self.embedder.get_embedding_info()
            embedded_chunks.append(chunk_copy)
        
        print(f"   ✅ Generated {len(embeddings)} CodeBERT embeddings")
        print(f"   📏 Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        print(f"   🤖 Model: {self.embedder.model_name}")
        
        return embedded_chunks
    
    def _detect_bridges(self, embedded_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 4: Detect theory-practice bridges."""
        print("   🔄 Detecting theory-practice bridges...")
        
        # Separate chunks by type
        python_chunks = [c for c in embedded_chunks if c['document_metadata']['file_type'] == 'python']
        text_chunks = [c for c in embedded_chunks if c['document_metadata']['file_type'] == 'text']
        
        print(f"   🔍 Analyzing {len(python_chunks)} Python vs {len(text_chunks)} text chunks")
        
        bridges = []
        
        # Keyword-based bridge detection
        theory_keywords = self.config.get('bridges', {}).get('theory_keywords', [])
        max_samples = self.config.get('bridges', {}).get('max_sample_chunks', 50)
        
        for text_chunk in text_chunks[:max_samples]:
            text_content = text_chunk['content'].lower()
            text_keywords = [kw for kw in theory_keywords if kw in text_content]
            
            if text_keywords:
                for python_chunk in python_chunks[:max_samples]:
                    python_content = python_chunk['content'].lower()
                    shared_keywords = [kw for kw in text_keywords if kw in python_content]
                    
                    if shared_keywords:
                        # Calculate bridge strength
                        strength = len(shared_keywords)
                        
                        # Add semantic similarity if embeddings available
                        if 'embedding' in text_chunk and 'embedding' in python_chunk:
                            similarity = self.embedder.calculate_similarity(
                                text_chunk['embedding'], 
                                python_chunk['embedding']
                            )
                            strength += similarity * self.config.get('bridges', {}).get('semantic_weight', 3.0)
                        
                        bridges.append({
                            'text_file': Path(text_chunk['metadata']['source_file']).name,
                            'python_file': Path(python_chunk['metadata']['source_file']).name,
                            'text_chunk_id': text_chunk['chunk_id'],
                            'python_chunk_id': python_chunk['chunk_id'],
                            'shared_concepts': shared_keywords,
                            'strength': strength,
                            'bridge_type': 'theory_practice'
                        })
        
        # Sort by strength
        bridges.sort(key=lambda x: x['strength'], reverse=True)
        
        print(f"   ✅ Detected {len(bridges)} theory-practice bridges")
        
        # Show top bridges
        for i, bridge in enumerate(bridges[:3]):
            print(f"      {i+1}. {bridge['text_file']} ↔ {bridge['python_file']} "
                  f"(strength: {bridge['strength']:.2f})")
        
        return bridges
    
    def _train_model(self, embedded_chunks: List[Dict[str, Any]]) -> tuple:
        """Phase 5: Train Sequential-ISNE model."""
        print("   🔄 Training Sequential-ISNE model...")
        
        # Convert to StreamingChunk objects
        streaming_chunks = []
        for chunk in embedded_chunks:
            metadata = ChunkMetadata(
                chunk_id=chunk['chunk_id'],
                chunk_type='content',  # Sequential-ISNE expects content chunks
                doc_path=chunk['metadata']['source_file'],
                directory=str(Path(chunk['metadata']['source_file']).parent),
                processing_order=chunk['chunk_id'],
                file_extension=Path(chunk['metadata']['source_file']).suffix
            )
            
            streaming_chunk = StreamingChunk(
                chunk_id=chunk['chunk_id'],
                content=chunk['content'],
                metadata=metadata,
                semantic_embedding=chunk['embedding']
            )
            streaming_chunks.append(streaming_chunk)
        
        # Generate relationships using streaming processor
        processor = StreamingChunkProcessor()
        processor.chunk_registry = {chunk.chunk_id: chunk for chunk in streaming_chunks}
        processor.current_chunk_id = len(streaming_chunks)
        
        for chunk in streaming_chunks:
            directory = chunk.metadata.directory
            processor.directory_chunks[directory].append(chunk.chunk_id)
        
        relationships = processor.get_sequential_relationships()
        
        print(f"   🔗 Generated {len(relationships)} sequential relationships")
        
        # Initialize and train model
        model = SequentialISNE(self.training_config)
        model.build_graph_from_chunks(streaming_chunks, relationships)
        
        print(f"   🏃 Training on {len(streaming_chunks)} chunks for {self.training_config.epochs} epochs...")
        training_results = model.train_embeddings()
        
        print(f"   ✅ Training completed")
        print(f"   📉 Final loss: {training_results.get('final_loss', 'N/A'):.6f}")
        print(f"   🎯 Epochs: {training_results.get('epochs_completed', 'N/A')}")
        
        return model, training_results
    
    def _validate_model(self, model, embedded_chunks: List[Dict[str, Any]], bridges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 6: Validate trained model."""
        print("   🔄 Validating Sequential-ISNE model...")
        
        validation_results = {}
        
        try:
            # Basic embedding quality
            if model.trained_embeddings is not None:
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity
                
                embeddings = model.trained_embeddings
                pairwise_sims = cosine_similarity(embeddings)
                avg_similarity = np.mean(pairwise_sims[np.triu_indices_from(pairwise_sims, k=1)])
                embedding_quality = 1.0 - min(avg_similarity, 0.95)
                
                validation_results['embedding_quality'] = embedding_quality
                validation_results['avg_embedding_similarity'] = avg_similarity
                
                print(f"   📊 Embedding quality: {embedding_quality:.3f}")
                print(f"   🎯 Avg similarity: {avg_similarity:.3f}")
            
            # Graph connectivity
            if hasattr(model, 'graph') and model.graph.graph.number_of_nodes() > 0:
                import networkx as nx
                
                graph = model.graph.graph
                is_connected = nx.is_weakly_connected(graph)
                avg_clustering = nx.average_clustering(graph.to_undirected())
                density = nx.density(graph)
                
                validation_results['graph_connected'] = is_connected
                validation_results['graph_clustering'] = avg_clustering
                validation_results['graph_density'] = density
                
                print(f"   🌐 Graph connected: {is_connected}")
                print(f"   🔗 Clustering coeff: {avg_clustering:.3f}")
                print(f"   📊 Density: {density:.3f}")
            
            # Bridge validation
            validation_results['bridges_detected'] = len(bridges)
            validation_results['bridge_validation'] = len(bridges) > 0
            
            print(f"   🌉 Bridges detected: {len(bridges)}")
            
        except Exception as e:
            print(f"   ⚠️ Validation error: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def _compile_results(self, normalized_docs, chunks, embedded_chunks, bridges, training_results, validation_results) -> Dict[str, Any]:
        """Phase 7: Compile complete results."""
        return {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'data_normalization': {
                'total_documents': len(normalized_docs),
                'file_types': self._count_file_types(normalized_docs),
                'normalization_method': 'docling'
            },
            'chunking': {
                'total_chunks': len(chunks),
                'chunking_stats': self.chunker.get_chunking_stats(chunks),
                'chunking_methods': ['chonky_text', 'ast_python']
            },
            'embedding': {
                'total_embeddings': len(embedded_chunks),
                'embedding_info': self.embedder.get_embedding_info(),
                'model_name': self.embedder.model_name
            },
            'bridges': {
                'total_detected': len(bridges),
                'detection_methods': ['keyword_matching', 'semantic_similarity'],
                'top_bridges': bridges[:5]
            },
            'training': training_results,
            'validation': validation_results,
            'pipeline_success': True
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        output_dir = Path(self.config.get('output', {}).get('results_directory', 'experiments'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"{self.experiment_name}_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=convert_numpy)
            print(f"   💾 Results saved: {results_file}")
        except Exception as e:
            print(f"   ⚠️ Failed to save results: {e}")
    
    def _display_summary(self, results: Dict[str, Any]):
        """Display final summary."""
        print(f"   📊 Documents: {results['data_normalization']['total_documents']}")
        print(f"   🔗 Chunks: {results['chunking']['total_chunks']}")
        print(f"   🧠 Embeddings: {results['embedding']['total_embeddings']}")
        print(f"   🌉 Bridges: {results['bridges']['total_detected']}")
        print(f"   🎯 Training epochs: {results['training'].get('epochs_completed', 'N/A')}")
        print(f"   ✅ Pipeline: {'SUCCESS' if results['pipeline_success'] else 'FAILED'}")
    
    def _count_file_types(self, docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count documents by file type."""
        counts = {}
        for doc in docs:
            file_type = doc['source']['file_type']
            counts[file_type] = counts.get(file_type, 0) + 1
        return counts


def main():
    """Run Sequential-ISNE demo from command line."""
    parser = argparse.ArgumentParser(
        description="Sequential-ISNE Complete Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py /path/to/dataset
    python demo.py /path/to/dataset --config custom_config.yaml
    
The dataset should contain:
    - PDF research papers (for theory content)
    - Python code files (for practice content)
    - Any text/markdown files
        """
    )
    
    parser.add_argument(
        'data_directory',
        help='Path to directory containing dataset (PDFs, code, docs)'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run demo
        demo = SequentialISNEDemo(args.config)
        results = demo.run_complete_pipeline(args.data_directory)
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📊 Check results in: experiments/{demo.experiment_name}_results.json")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())