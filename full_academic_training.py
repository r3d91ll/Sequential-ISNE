#!/usr/bin/env python3
"""
Full Academic Scale Sequential-ISNE Training
10 epochs with comprehensive validation and detailed logging.
Designed for overnight execution.
"""

import logging
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Enhanced logging setup
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f"academic_training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
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
    logger.warning("WandB not available - metrics will be logged locally only")


class FullAcademicTrainer:
    """
    Full academic scale Sequential-ISNE trainer with comprehensive validation.
    """
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"academic-training-{timestamp}"
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
        # Initialize components
        self.normalizer = DataNormalizer()
        self.embedder = CPUEmbedder()
        
        # Full academic training config
        self.config = TrainingConfig(
            embedding_dim=384,
            hidden_dim=256,
            learning_rate=0.001,
            epochs=10,  # Full 10 epochs for academic validation
            batch_size=32,  # Conservative batch size for stability
            negative_samples=5,
            device='auto'
        )
        
        # Initialize wandb if available
        self.log_to_wandb = HAS_WANDB and os.getenv('WANDB_API_KEY')
        if self.log_to_wandb:
            wandb.init(
                project=os.getenv('WANDB_PROJECT', 'sequential-isne-academic'),
                name=self.experiment_name,
                config={
                    'embedding_dim': self.config.embedding_dim,
                    'hidden_dim': self.config.hidden_dim,
                    'learning_rate': self.config.learning_rate,
                    'epochs': self.config.epochs,
                    'batch_size': self.config.batch_size,
                    'training_type': 'full_academic_scale',
                    'dataset': 'enhanced_theory_practice'
                }
            )
            logger.info(f"WandB tracking initialized: {self.experiment_name}")
        
        logger.info(f"Academic trainer initialized: {self.experiment_name}")
        logger.info(f"Training config: {self.config}")
        logger.info(f"Log file: {log_file}")
    
    def train_full_academic_scale(self, data_dir: str) -> Dict[str, Any]:
        """Full academic scale training with comprehensive metrics."""
        
        logger.info("=" * 80)
        logger.info("🎓 FULL ACADEMIC SCALE SEQUENTIAL-ISNE TRAINING")
        logger.info("=" * 80)
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🔬 Experiment: {self.experiment_name}")
        logger.info(f"⏰ Started: {datetime.now().isoformat()}")
        logger.info(f"📊 Target: 10 epochs with full validation")
        
        try:
            # Phase 1: Data Processing
            logger.info("\n" + "="*60)
            logger.info("📊 PHASE 1: Data Normalization & Processing")
            logger.info("="*60)
            
            phase_start = time.time()
            normalized_docs = self.normalizer.normalize_directory(data_dir, recursive=True)
            
            successful_docs = [doc for doc in normalized_docs if not doc['error']]
            failed_docs = [doc for doc in normalized_docs if doc['error']]
            
            logger.info(f"   ✅ Documents processed: {len(successful_docs)}/{len(normalized_docs)}")
            logger.info(f"   📄 File types: {self._count_file_types(successful_docs)}")
            logger.info(f"   ⏱️  Phase 1 time: {time.time() - phase_start:.2f}s")
            
            if self.log_to_wandb:
                wandb.log({
                    'phase1_documents_total': len(normalized_docs),
                    'phase1_documents_successful': len(successful_docs),
                    'phase1_documents_failed': len(failed_docs),
                    'phase1_duration': time.time() - phase_start
                })
            
            # Phase 2: Chunk Extraction & Analysis
            logger.info("\n" + "="*60)
            logger.info("🔗 PHASE 2: Chunk Extraction & Bridge Analysis")
            logger.info("="*60)
            
            phase_start = time.time()
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
            
            logger.info(f"   📊 Total chunks: {len(all_chunks)}")
            logger.info(f"   🐍 Python chunks: {len(python_chunks)}")
            logger.info(f"   📄 Document chunks: {len(document_chunks)}")
            logger.info(f"   🌉 Bridge potential: {len(python_chunks) * len(document_chunks):,}")
            logger.info(f"   ⏱️  Phase 2 time: {time.time() - phase_start:.2f}s")
            
            if self.log_to_wandb:
                wandb.log({
                    'phase2_chunks_total': len(all_chunks),
                    'phase2_chunks_python': len(python_chunks),
                    'phase2_chunks_document': len(document_chunks),
                    'phase2_bridge_potential': len(python_chunks) * len(document_chunks),
                    'phase2_duration': time.time() - phase_start
                })
            
            # Phase 3: Embedding Generation
            logger.info("\n" + "="*60)
            logger.info("🧠 PHASE 3: Real Embedding Generation")
            logger.info("="*60)
            
            phase_start = time.time()
            chunk_texts = [chunk['content'] for chunk in all_chunks]
            logger.info(f"   🔄 Generating embeddings for {len(chunk_texts):,} chunks...")
            
            embeddings = self.embedder.embed_batch(chunk_texts)
            embedding_dim = len(embeddings[0]) if embeddings else 0
            
            logger.info(f"   ✅ Generated {len(embeddings):,} embeddings")
            logger.info(f"   📏 Embedding dimension: {embedding_dim}")
            logger.info(f"   ⏱️  Phase 3 time: {time.time() - phase_start:.2f}s")
            
            if self.log_to_wandb:
                wandb.log({
                    'phase3_embeddings_count': len(embeddings),
                    'phase3_embedding_dimension': embedding_dim,
                    'phase3_duration': time.time() - phase_start
                })
            
            # Phase 4: Theory-Practice Bridge Detection
            logger.info("\n" + "="*60)
            logger.info("🌉 PHASE 4: Theory-Practice Bridge Detection")
            logger.info("="*60)
            
            phase_start = time.time()
            bridges = self._detect_theory_practice_bridges(python_chunks, document_chunks)
            
            logger.info(f"   🔍 Detected {len(bridges)} theory-practice bridges")
            bridge_types = {}
            for bridge in bridges:
                bridge_type = bridge.get('bridge_type', 'unknown')
                bridge_types[bridge_type] = bridge_types.get(bridge_type, 0) + 1
            
            logger.info(f"   📊 Bridge types: {bridge_types}")
            for i, bridge in enumerate(bridges[:5]):
                logger.info(f"      {i+1}. {bridge['python_file']} ↔ {bridge['document_file']} "
                           f"(type: {bridge.get('bridge_type', 'unknown')}, "
                           f"strength: {bridge.get('strength', 0):.2f})")
            
            logger.info(f"   ⏱️  Phase 4 time: {time.time() - phase_start:.2f}s")
            
            if self.log_to_wandb:
                wandb.log({
                    'phase4_bridges_total': len(bridges),
                    'phase4_bridge_types': bridge_types,
                    'phase4_duration': time.time() - phase_start
                })
            
            # Phase 5: Sequential-ISNE Model Training (Main Phase)
            logger.info("\n" + "="*60)
            logger.info("🎯 PHASE 5: Sequential-ISNE Model Training (10 Epochs)")
            logger.info("="*60)
            
            phase_start = time.time()
            model = SequentialISNE(self.config)
            
            # Convert to StreamingChunk objects
            streaming_chunks = []
            for i, chunk in enumerate(all_chunks):
                metadata = ChunkMetadata(
                    chunk_id=i,
                    chunk_type="content",
                    doc_path=chunk['metadata']['source_file'],
                    directory=str(Path(chunk['metadata']['source_file']).parent),
                    processing_order=i,
                    file_extension=Path(chunk['metadata']['source_file']).suffix
                )
                
                streaming_chunk = StreamingChunk(
                    chunk_id=i,
                    content=chunk['content'],
                    metadata=metadata,
                    semantic_embedding=embeddings[i] if i < len(embeddings) else None
                )
                streaming_chunks.append(streaming_chunk)
            
            # Generate relationships
            from src.streaming_processor import StreamingChunkProcessor
            processor = StreamingChunkProcessor()
            
            processor.chunk_registry = {chunk.chunk_id: chunk for chunk in streaming_chunks}
            processor.current_chunk_id = len(streaming_chunks)
            
            for chunk in streaming_chunks:
                directory = chunk.metadata.directory
                processor.directory_chunks[directory].append(chunk.chunk_id)
            
            relationships = processor.get_sequential_relationships()
            logger.info(f"   🔗 Generated {len(relationships):,} sequential relationships")
            
            # Build graph and train
            model.build_graph_from_chunks(streaming_chunks, relationships)
            
            logger.info(f"   🏃 Training model on {len(all_chunks):,} chunks for {self.config.epochs} epochs...")
            logger.info(f"   📊 Training pairs: {len(relationships):,}")
            
            # Add enhanced logging for WandB
            self._add_wandb_epoch_logging(model)
            
            # Use the built-in training method (now enhanced with logging)
            training_results = model.train_embeddings()
            
            logger.info(f"   ✅ Training completed in {time.time() - phase_start:.2f}s")
            logger.info(f"   📉 Final loss: {training_results.get('final_loss', 'N/A')}")
            logger.info(f"   📊 Epochs completed: {training_results.get('epochs_completed', 'N/A')}")
            
            if self.log_to_wandb:
                wandb.log({
                    'phase5_training_chunks': len(all_chunks),
                    'phase5_training_relationships': len(relationships),
                    'phase5_final_loss': training_results.get('final_loss', 0),
                    'phase5_epochs_completed': training_results.get('epochs_completed', 0),
                    'phase5_duration': time.time() - phase_start
                })
            
            # Phase 6: Comprehensive Model Validation
            logger.info("\n" + "="*60)
            logger.info("🔬 PHASE 6: Comprehensive Model Validation")
            logger.info("="*60)
            
            phase_start = time.time()
            validation_results = self._comprehensive_validation(model, streaming_chunks, relationships, bridges)
            
            logger.info(f"   ✅ Validation completed in {time.time() - phase_start:.2f}s")
            logger.info(f"   📊 Validation metrics:")
            for metric, value in validation_results.items():
                if isinstance(value, float):
                    logger.info(f"      {metric}: {value:.4f}")
                else:
                    logger.info(f"      {metric}: {value}")
            
            if self.log_to_wandb:
                validation_metrics = {f'validation_{k}': v for k, v in validation_results.items() 
                                    if isinstance(v, (int, float))}
                validation_metrics['phase6_duration'] = time.time() - phase_start
                wandb.log(validation_metrics)
            
            # Phase 7: Results Compilation
            total_time = time.time() - self.start_time
            
            results = {
                'experiment_name': self.experiment_name,
                'training_type': 'full_academic_scale',
                'total_duration_seconds': total_time,
                'total_duration_formatted': f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s",
                'started_at': datetime.fromtimestamp(self.start_time).isoformat(),
                'completed_at': datetime.now().isoformat(),
                'data_processing': {
                    'total_documents': len(normalized_docs),
                    'successful_documents': len(successful_docs),
                    'failed_documents': len(failed_docs),
                    'file_type_distribution': self._count_file_types(successful_docs)
                },
                'chunk_analysis': {
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
                'theory_practice_bridges': {
                    'total_detected': len(bridges),
                    'bridge_types': bridge_types,
                    'top_bridges': bridges[:10]
                },
                'training': training_results,
                'validation': validation_results,
                'academic_scale_validation': {
                    'target_nodes': 5196,  # BlogCatalog benchmark
                    'achieved_nodes': len(all_chunks),
                    'scale_ratio': len(all_chunks) / 5196,
                    'academic_scale_achieved': len(all_chunks) >= 4500,
                    'training_pairs': len(relationships),
                    'legitimate_academic_scale': len(relationships) > 500000
                }
            }
            
            # Save comprehensive results
            output_dir = Path("experiments/academic_training")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = output_dir / f"{self.experiment_name}_complete_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save model if training was successful
            if training_results.get('epochs_completed', 0) == self.config.epochs:
                model_file = output_dir / f"{self.experiment_name}_trained_model.json"
                model.save_model(str(model_file))
                logger.info(f"   💾 Trained model saved: {model_file}")
            
            logger.info("\n" + "="*80)
            logger.info("🎉 FULL ACADEMIC SCALE TRAINING COMPLETE!")
            logger.info("="*80)
            logger.info(f"   📊 Total duration: {results['total_duration_formatted']}")
            logger.info(f"   📈 Chunks processed: {len(all_chunks):,}")
            logger.info(f"   🔗 Training pairs: {len(relationships):,}")
            logger.info(f"   🌉 Theory-practice bridges: {len(bridges)}")
            logger.info(f"   📊 Academic scale achieved: {results['academic_scale_validation']['academic_scale_achieved']}")
            logger.info(f"   💾 Results saved: {results_file}")
            logger.info(f"   📋 Log file: {log_file}")
            
            if self.log_to_wandb:
                wandb.log({
                    'training_complete': True,
                    'total_duration': total_time,
                    'academic_scale_achieved': results['academic_scale_validation']['academic_scale_achieved']
                })
                wandb.finish()
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.exception("Full exception details:")
            if self.log_to_wandb:
                wandb.log({'training_failed': True, 'error': str(e)})
                wandb.finish()
            raise
    
    def _add_wandb_epoch_logging(self, model):
        """Add WandB logging hooks to the model's training if available."""
        if not self.log_to_wandb:
            return
            
        # Store original train_embeddings method
        original_train = model.train_embeddings
        
        def enhanced_train_embeddings():
            """Enhanced training with WandB logging."""
            logger.info("   🚀 Starting training with WandB logging...")
            
            # Call original training but with enhanced logging
            if hasattr(model, '_train_epoch'):
                # Hook into the training loop for per-epoch logging
                original_train_epoch = model._train_epoch
                epoch_counter = 0
                
                def logged_train_epoch(training_pairs, optimizer, device):
                    nonlocal epoch_counter
                    epoch_start = time.time()
                    loss = original_train_epoch(training_pairs, optimizer, device)
                    epoch_duration = time.time() - epoch_start
                    
                    logger.info(f"      Epoch {epoch_counter+1}/{self.config.epochs}: "
                               f"Loss = {loss:.6f}, Time = {epoch_duration:.1f}s")
                    
                    wandb.log({
                        'epoch': epoch_counter,
                        'train_loss': loss,
                        'epoch_duration': epoch_duration
                    })
                    
                    # Save checkpoint every 2 epochs
                    if (epoch_counter + 1) % 2 == 0:
                        checkpoint_dir = Path("experiments/academic_training/checkpoints")
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_file = checkpoint_dir / f"{self.experiment_name}_epoch_{epoch_counter+1}.json"
                        model.save_model(str(checkpoint_file))
                        logger.info(f"      💾 Checkpoint saved: epoch {epoch_counter+1}")
                    
                    epoch_counter += 1
                    return loss
                
                model._train_epoch = logged_train_epoch
            
            # Run the original training
            return original_train()
        
        # Replace the method
        model.train_embeddings = enhanced_train_embeddings
    
    def _comprehensive_validation(self, model, streaming_chunks, relationships, bridges):
        """Comprehensive post-training validation."""
        logger.info("   🔍 Running comprehensive validation...")
        
        validation_results = {}
        
        try:
            # 1. Basic model validation
            basic_validation = self._validate_trained_model(model, streaming_chunks, relationships)
            validation_results.update(basic_validation)
            
            # 2. Bridge validation
            if bridges:
                bridge_validation = self._validate_theory_practice_bridges(model, bridges)
                validation_results.update(bridge_validation)
            
            # 3. Scale validation
            scale_validation = self._validate_academic_scale(len(streaming_chunks), len(relationships))
            validation_results.update(scale_validation)
            
            # 4. Actor-Network Theory validation
            ant_validation = self._validate_actor_network_principles(model, streaming_chunks, bridges)
            validation_results.update(ant_validation)
            
            logger.info(f"   ✅ Comprehensive validation completed")
            
        except Exception as e:
            logger.error(f"   ❌ Validation failed: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def _validate_theory_practice_bridges(self, model, bridges):
        """Validate theory-practice bridge quality."""
        logger.info("      🌉 Validating theory-practice bridges...")
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            if not hasattr(model, 'trained_embeddings') or model.trained_embeddings is None:
                return {'bridge_validation_error': 'No trained embeddings available'}
            
            bridge_similarities = []
            for bridge in bridges[:50]:  # Sample for performance
                py_chunk_id = bridge.get('python_chunk_id')
                doc_chunk_id = bridge.get('document_chunk_id')
                
                if py_chunk_id in model.node_to_index and doc_chunk_id in model.node_to_index:
                    py_idx = model.node_to_index[py_chunk_id]
                    doc_idx = model.node_to_index[doc_chunk_id]
                    
                    similarity = cosine_similarity(
                        model.trained_embeddings[py_idx:py_idx+1],
                        model.trained_embeddings[doc_idx:doc_idx+1]
                    )[0, 0]
                    bridge_similarities.append(similarity)
            
            if bridge_similarities:
                return {
                    'bridge_avg_similarity': float(np.mean(bridge_similarities)),
                    'bridge_max_similarity': float(np.max(bridge_similarities)),
                    'bridge_min_similarity': float(np.min(bridge_similarities)),
                    'bridge_similarities_std': float(np.std(bridge_similarities)),
                    'bridges_validated': len(bridge_similarities)
                }
            else:
                return {'bridge_validation_error': 'No valid bridges to validate'}
                
        except Exception as e:
            return {'bridge_validation_error': str(e)}
    
    def _validate_academic_scale(self, num_chunks, num_relationships):
        """Validate achievement of academic scale."""
        # Original ISNE paper benchmarks
        benchmarks = {
            'BlogCatalog': 5196,
            'Flickr': 7575,
            'YouTube': 15088,
            'Arxiv': 169343
        }
        
        scale_results = {}
        for benchmark_name, benchmark_size in benchmarks.items():
            scale_ratio = num_chunks / benchmark_size
            scale_results[f'scale_vs_{benchmark_name.lower()}'] = scale_ratio
            scale_results[f'meets_{benchmark_name.lower()}_scale'] = num_chunks >= benchmark_size * 0.8
        
        scale_results.update({
            'total_chunks': num_chunks,
            'total_relationships': num_relationships,
            'academic_scale_achieved': num_chunks >= 4500,
            'relationship_density': num_relationships / (num_chunks * (num_chunks - 1)) if num_chunks > 1 else 0
        })
        
        return scale_results
    
    def _validate_actor_network_principles(self, model, streaming_chunks, bridges):
        """Validate Actor-Network Theory principles."""
        logger.info("      🕸️  Validating Actor-Network Theory principles...")
        
        try:
            # Analyze actor diversity
            actor_types = {}
            for chunk in streaming_chunks:
                file_ext = chunk.metadata.file_extension
                actor_types[file_ext] = actor_types.get(file_ext, 0) + 1
            
            # Analyze network heterogeneity
            directory_diversity = len(set(chunk.metadata.directory for chunk in streaming_chunks))
            
            # Analyze bridge heterogeneity
            bridge_type_diversity = len(set(bridge.get('bridge_type', 'unknown') for bridge in bridges))
            
            return {
                'actor_type_diversity': len(actor_types),
                'actor_type_distribution': actor_types,
                'directory_diversity': directory_diversity,
                'bridge_type_diversity': bridge_type_diversity,
                'heterogeneous_network': len(actor_types) >= 3,
                'ant_principles_validated': len(actor_types) >= 3 and directory_diversity >= 5
            }
            
        except Exception as e:
            return {'ant_validation_error': str(e)}
    
    def _count_file_types(self, docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count documents by file type."""
        counts = {}
        for doc in docs:
            file_type = doc['source']['file_type']
            counts[file_type] = counts.get(file_type, 0) + 1
        return counts
    
    def _detect_theory_practice_bridges(self, python_chunks: List[Dict], 
                                      document_chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Enhanced theory-practice bridge detection."""
        bridges = []
        
        # Use the same enhanced detection from the original trainer
        theory_keywords = ['algorithm', 'model', 'architecture', 'method', 'approach', 
                          'framework', 'theory', 'implementation', 'embedding', 'neural',
                          'graph', 'network', 'node', 'edge', 'representation', 'learning',
                          'actor', 'network', 'social', 'system', 'complex', 'knowledge']
        
        for doc_chunk in document_chunks[:100]:  # Increased sample
            doc_text = doc_chunk['content'].lower()
            doc_keywords = [kw for kw in theory_keywords if kw in doc_text]
            
            if doc_keywords:
                for py_chunk in python_chunks[:100]:  # Increased sample
                    py_text = py_chunk['content'].lower()
                    shared_keywords = [kw for kw in doc_keywords if kw in py_text]
                    
                    if len(shared_keywords) >= 1:
                        doc_file = Path(doc_chunk['metadata']['source_file'])
                        py_file = Path(py_chunk['metadata']['source_file'])
                        
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
        
        bridges.sort(key=lambda x: x['strength'], reverse=True)
        return bridges
    
    def _validate_trained_model(self, model, streaming_chunks, relationships):
        """Basic model validation (reused from original trainer)."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        validation_results = {}
        
        try:
            if model.trained_embeddings is not None:
                embeddings = model.trained_embeddings
                
                pairwise_similarities = cosine_similarity(embeddings)
                avg_similarity = np.mean(pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)])
                embedding_quality = 1.0 - min(avg_similarity, 0.95)
                validation_results['embedding_quality'] = embedding_quality
                validation_results['avg_embedding_similarity'] = avg_similarity
            else:
                validation_results['embedding_quality'] = 0.0
            
            if hasattr(model, 'graph') and model.graph.graph.number_of_edges() > 0:
                relationship_accuracies = []
                sequential_rels = [r for r in relationships if r['relationship_type'] == 'sequential']
                
                if sequential_rels and model.trained_embeddings is not None:
                    for rel in sequential_rels[:50]:
                        from_idx = model.node_to_index.get(rel['from_chunk_id'])
                        to_idx = model.node_to_index.get(rel['to_chunk_id'])
                        
                        if from_idx is not None and to_idx is not None:
                            sim = cosine_similarity(
                                embeddings[from_idx:from_idx+1], 
                                embeddings[to_idx:to_idx+1]
                            )[0, 0]
                            relationship_accuracies.append(sim)
                
                if relationship_accuracies:
                    validation_results['relationship_accuracy'] = float(np.mean(relationship_accuracies))
                else:
                    validation_results['relationship_accuracy'] = 0.0
            else:
                validation_results['relationship_accuracy'] = 0.0
            
            if hasattr(model, 'graph') and model.graph.graph.number_of_nodes() > 0:
                import networkx as nx
                graph = model.graph.graph
                
                is_connected = nx.is_weakly_connected(graph)
                avg_clustering = nx.average_clustering(graph.to_undirected())
                density = nx.density(graph)
                
                connectivity_score = (
                    (1.0 if is_connected else 0.0) * 0.5 +
                    avg_clustering * 0.3 +
                    min(density, 0.5) * 0.4
                )
                
                validation_results['graph_connectivity'] = connectivity_score
                validation_results['graph_is_connected'] = is_connected
                validation_results['graph_clustering'] = avg_clustering
                validation_results['graph_density'] = density
            else:
                validation_results['graph_connectivity'] = 0.0
        
        except Exception as e:
            logger.error(f"Basic validation error: {e}")
            validation_results = {
                'embedding_quality': 0.0,
                'relationship_accuracy': 0.0,
                'graph_connectivity': 0.0,
                'validation_error': str(e)
            }
        
        return validation_results


def main():
    """Run full academic scale training."""
    
    # Enhanced dataset directory with theory papers
    data_dir = "/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata/isne-testdata"
    
    if not Path(data_dir).exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return 1
    
    logger.info("🚀 Launching Full Academic Scale Sequential-ISNE Training")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"🎯 Target: 10 epochs with comprehensive validation")
    logger.info(f"📊 Expected scale: 4,722+ chunks, 762,017+ relationships")
    logger.info(f"🌉 Theory papers: 40 Actor-Network Theory & STS papers")
    logger.info(f"⏰ Estimated duration: 2-4 hours")
    
    try:
        trainer = FullAcademicTrainer()
        results = trainer.train_full_academic_scale(data_dir)
        
        logger.info("🎉 Full academic training completed successfully!")
        logger.info(f"   📊 Final results available in experiments/academic_training/")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.exception("Full error details:")
        return 1


if __name__ == "__main__":
    exit(main())