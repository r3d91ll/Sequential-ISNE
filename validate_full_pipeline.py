#!/usr/bin/env python3
"""
Comprehensive validation of Sequential-ISNE pipeline with data density analysis.

This script validates:
1. Data collection and density
2. Document processing effectiveness
3. Chunk creation and relationships
4. Model training convergence
5. Embedding quality
6. Theory-practice bridge discovery
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict

# Import components
from src.streaming_processor import StreamingChunkProcessor
from src.sequential_isne import SequentialISNE, TrainingConfig
from src.simple_document_processor import SimpleDocumentProcessor
from src.chunker import TextChunker
from src.cpu_embedder import CPUEmbedder
from src.doc_types import ChunkingInput


class ComprehensivePipelineValidator:
    """Validates the complete Sequential-ISNE pipeline with detailed metrics."""
    
    def __init__(self, test_data_dir: str = "/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata"):
        self.test_data_dir = Path(test_data_dir)
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_metrics": {},
            "processing_metrics": {},
            "training_metrics": {},
            "validation_scores": {}
        }
        
    def validate_data_density(self) -> Dict[str, Any]:
        """Analyze data density and collection metrics."""
        print("\n🔍 Analyzing Data Density...")
        
        metrics = {
            "total_files": 0,
            "files_by_type": defaultdict(int),
            "total_size_bytes": 0,
            "repositories": defaultdict(dict),
            "pdf_files": [],
            "code_files": []
        }
        
        # Scan test data
        for repo_dir in self.test_data_dir.iterdir():
            if not repo_dir.is_dir():
                continue
                
            repo_name = repo_dir.name
            repo_files = list(repo_dir.rglob("*"))
            file_count = len([f for f in repo_files if f.is_file()])
            
            metrics["repositories"][repo_name] = {
                "file_count": file_count,
                "total_size": 0,
                "file_types": defaultdict(int)
            }
            
            for file_path in repo_files:
                if file_path.is_file():
                    metrics["total_files"] += 1
                    file_size = file_path.stat().st_size
                    metrics["total_size_bytes"] += file_size
                    metrics["repositories"][repo_name]["total_size"] += file_size
                    
                    ext = file_path.suffix.lower()
                    metrics["files_by_type"][ext] += 1
                    metrics["repositories"][repo_name]["file_types"][ext] += 1
                    
                    if ext == ".pdf":
                        metrics["pdf_files"].append(str(file_path))
                    elif ext in [".py", ".js", ".java", ".cpp", ".c"]:
                        metrics["code_files"].append(str(file_path))
        
        # Calculate density metrics
        metrics["total_size_mb"] = round(metrics["total_size_bytes"] / (1024 * 1024), 2)
        metrics["avg_file_size_kb"] = round(metrics["total_size_bytes"] / max(metrics["total_files"], 1) / 1024, 2)
        metrics["pdf_to_code_ratio"] = len(metrics["pdf_files"]) / max(len(metrics["code_files"]), 1)
        
        print(f"   📁 Total files: {metrics['total_files']}")
        print(f"   💾 Total size: {metrics['total_size_mb']} MB")
        print(f"   📊 Repositories: {len(metrics['repositories'])}")
        print(f"   📄 PDF files: {len(metrics['pdf_files'])}")
        print(f"   🐍 Code files: {len(metrics['code_files'])}")
        
        self.results["data_metrics"] = metrics
        return metrics
    
    def validate_processing_pipeline(self) -> Dict[str, Any]:
        """Test document processing with detailed metrics."""
        print("\n⚙️ Validating Document Processing Pipeline...")
        
        processor = SimpleDocumentProcessor()
        chunker = TextChunker()
        
        processing_metrics = {
            "documents_processed": 0,
            "processing_failures": 0,
            "total_chunks": 0,
            "chunks_by_type": defaultdict(int),
            "processing_times": [],
            "chunk_sizes": [],
            "pdf_extraction_success": 0
        }
        
        # Process all files
        all_files = []
        for repo_dir in self.test_data_dir.iterdir():
            if repo_dir.is_dir():
                all_files.extend(repo_dir.rglob("*"))
        
        for file_path in all_files:
            if not file_path.is_file():
                continue
                
            start_time = time.time()
            try:
                # Process document
                doc = processor.process_document(file_path)
                
                if doc.error:
                    processing_metrics["processing_failures"] += 1
                else:
                    processing_metrics["documents_processed"] += 1
                    
                    if file_path.suffix.lower() == ".pdf":
                        processing_metrics["pdf_extraction_success"] += 1
                    
                    # Chunk the content
                    chunking_input = ChunkingInput(
                        text=doc.content,
                        document_id=str(file_path),
                        chunk_size=512,
                        chunk_overlap=50
                    )
                    chunking_output = chunker.chunk(chunking_input)
                    chunks = chunking_output.chunks
                    
                    processing_metrics["total_chunks"] += len(chunks)
                    processing_metrics["chunks_by_type"][doc.content_type] += len(chunks)
                    
                    for chunk in chunks:
                        processing_metrics["chunk_sizes"].append(len(chunk.text))
                
                processing_time = time.time() - start_time
                processing_metrics["processing_times"].append(processing_time)
                
            except Exception as e:
                print(f"   ❌ Failed to process {file_path}: {e}")
                processing_metrics["processing_failures"] += 1
        
        # Calculate statistics
        if processing_metrics["processing_times"]:
            processing_metrics["avg_processing_time"] = np.mean(processing_metrics["processing_times"])
            processing_metrics["total_processing_time"] = sum(processing_metrics["processing_times"])
        
        if processing_metrics["chunk_sizes"]:
            processing_metrics["avg_chunk_size"] = np.mean(processing_metrics["chunk_sizes"])
            processing_metrics["chunk_size_std"] = np.std(processing_metrics["chunk_sizes"])
        
        print(f"   ✅ Documents processed: {processing_metrics['documents_processed']}")
        print(f"   ❌ Processing failures: {processing_metrics['processing_failures']}")
        print(f"   📊 Total chunks: {processing_metrics['total_chunks']}")
        print(f"   ⏱️ Avg processing time: {processing_metrics.get('avg_processing_time', 0):.3f}s")
        
        self.results["processing_metrics"] = processing_metrics
        return processing_metrics
    
    def validate_model_training(self) -> Dict[str, Any]:
        """Validate model training with convergence analysis."""
        print("\n🧠 Validating Model Training...")
        
        # Create streaming processor
        processor = StreamingChunkProcessor()
        
        # Process test data
        all_files = []
        for repo_dir in self.test_data_dir.iterdir():
            if repo_dir.is_dir():
                all_files.extend(str(f) for f in repo_dir.rglob("*") if f.is_file())
        
        chunks = list(processor.process_documents(all_files))
        content_chunks = [c for c in chunks if c.metadata.chunk_type == "content"]
        
        print(f"   📊 Content chunks for training: {len(content_chunks)}")
        
        # Initialize model with config
        config = TrainingConfig(
            embedding_dim=384,
            hidden_dim=256,
            learning_rate=0.001,
            epochs=100,
            batch_size=32
        )
        model = SequentialISNE(config)
        
        # Track training metrics
        training_metrics = {
            "total_chunks": len(content_chunks),
            "training_chunks": len(content_chunks),
            "epochs": 100,  # More epochs for better convergence analysis
            "losses": [],
            "convergence_rate": 0,
            "final_loss": 0,
            "training_time": 0
        }
        
        # Add chunks to model for relationship discovery
        for chunk in content_chunks:
            model.add_chunk(
                chunk_id=chunk.metadata.chunk_id,
                content=chunk.text,
                metadata=chunk.metadata.__dict__
            )
        
        print("   🏃 Training model...")
        start_time = time.time()
        
        # Train the model
        training_results = model.train_embeddings()
        
        training_time = time.time() - start_time
        training_metrics["training_time"] = training_time
        training_metrics["final_loss"] = training_results.get("final_loss", 0)
        training_metrics["losses"] = training_results.get("losses", [])
        
        # Calculate convergence rate
        if len(training_metrics["losses"]) > 10:
            early_losses = training_metrics["losses"][:10]
            late_losses = training_metrics["losses"][-10:]
            training_metrics["convergence_rate"] = (np.mean(early_losses) - np.mean(late_losses)) / np.mean(early_losses)
        
        print(f"   ⏱️ Training time: {training_time:.2f}s")
        print(f"   📉 Final loss: {training_metrics['final_loss']:.4f}")
        print(f"   📈 Convergence rate: {training_metrics['convergence_rate']:.2%}")
        
        self.results["training_metrics"] = training_metrics
        return training_metrics
    
    def validate_theory_practice_bridges(self) -> Dict[str, Any]:
        """Validate theory-practice bridge discovery."""
        print("\n🌉 Validating Theory-Practice Bridges...")
        
        bridge_metrics = {
            "pdf_chunks": 0,
            "code_chunks": 0,
            "potential_bridges": 0,
            "strong_bridges": 0,
            "bridge_examples": []
        }
        
        # Simple heuristic: count PDF chunks that share concepts with code chunks
        processor = SimpleDocumentProcessor()
        chunker = TextChunker()
        
        pdf_chunks = []
        code_chunks = []
        
        for repo_dir in self.test_data_dir.iterdir():
            if not repo_dir.is_dir():
                continue
                
            for file_path in repo_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                    
                try:
                    doc = processor.process_document(file_path)
                    if not doc.error:
                        chunking_input = ChunkingInput(
                            text=doc.content,
                            document_id=str(file_path)
                        )
                        chunking_output = chunker.chunk(chunking_input)
                        chunks = chunking_output.chunks
                        
                        if file_path.suffix.lower() == ".pdf":
                            pdf_chunks.extend([(c, file_path) for c in chunks])
                        elif file_path.suffix.lower() in [".py", ".js"]:
                            code_chunks.extend([(c, file_path) for c in chunks])
                except:
                    pass
        
        bridge_metrics["pdf_chunks"] = len(pdf_chunks)
        bridge_metrics["code_chunks"] = len(code_chunks)
        
        # Find potential bridges (simplified concept matching)
        concept_keywords = ["embedding", "graph", "node", "isne", "pathrag", "retrieval", "augmented"]
        
        for pdf_chunk, pdf_file in pdf_chunks[:10]:  # Limit for performance
            pdf_text_lower = pdf_chunk.text.lower()
            pdf_concepts = [kw for kw in concept_keywords if kw in pdf_text_lower]
            
            if pdf_concepts:
                for code_chunk, code_file in code_chunks[:20]:  # Limit for performance
                    code_text_lower = code_chunk.text.lower()
                    shared_concepts = [kw for kw in pdf_concepts if kw in code_text_lower]
                    
                    if shared_concepts:
                        bridge_metrics["potential_bridges"] += 1
                        
                        if len(shared_concepts) >= 2:
                            bridge_metrics["strong_bridges"] += 1
                            
                            if len(bridge_metrics["bridge_examples"]) < 3:
                                bridge_metrics["bridge_examples"].append({
                                    "pdf_file": pdf_file.name,
                                    "code_file": code_file.name,
                                    "shared_concepts": shared_concepts,
                                    "pdf_snippet": pdf_chunk.text[:100],
                                    "code_snippet": code_chunk.text[:100]
                                })
        
        print(f"   📄 PDF chunks: {bridge_metrics['pdf_chunks']}")
        print(f"   💻 Code chunks: {bridge_metrics['code_chunks']}")
        print(f"   🔗 Potential bridges: {bridge_metrics['potential_bridges']}")
        print(f"   💪 Strong bridges: {bridge_metrics['strong_bridges']}")
        
        self.results["validation_scores"]["bridge_metrics"] = bridge_metrics
        return bridge_metrics
    
    def calculate_overall_validation(self) -> Dict[str, Any]:
        """Calculate overall validation scores."""
        print("\n📊 Calculating Overall Validation...")
        
        scores = {
            "data_sufficiency": 0,
            "processing_effectiveness": 0,
            "training_convergence": 0,
            "bridge_discovery": 0,
            "overall_score": 0
        }
        
        # Data sufficiency score
        data_metrics = self.results["data_metrics"]
        if data_metrics["total_files"] >= 20 and data_metrics["total_size_mb"] >= 1:
            scores["data_sufficiency"] = min(100, (data_metrics["total_files"] / 20) * 50 + 
                                            (data_metrics["total_size_mb"] / 10) * 50)
        
        # Processing effectiveness
        proc_metrics = self.results["processing_metrics"]
        if proc_metrics["documents_processed"] > 0:
            success_rate = proc_metrics["documents_processed"] / (
                proc_metrics["documents_processed"] + proc_metrics["processing_failures"]
            )
            scores["processing_effectiveness"] = success_rate * 100
        
        # Training convergence
        train_metrics = self.results["training_metrics"]
        if train_metrics.get("convergence_rate", 0) > 0:
            scores["training_convergence"] = min(100, train_metrics["convergence_rate"] * 200)
        
        # Bridge discovery
        bridge_metrics = self.results["validation_scores"].get("bridge_metrics", {})
        if bridge_metrics.get("pdf_chunks", 0) > 0 and bridge_metrics.get("code_chunks", 0) > 0:
            bridge_ratio = bridge_metrics.get("potential_bridges", 0) / (
                min(bridge_metrics["pdf_chunks"], bridge_metrics["code_chunks"]) + 1
            )
            scores["bridge_discovery"] = min(100, bridge_ratio * 100)
        
        # Overall score
        scores["overall_score"] = np.mean([
            scores["data_sufficiency"],
            scores["processing_effectiveness"],
            scores["training_convergence"],
            scores["bridge_discovery"]
        ])
        
        self.results["validation_scores"]["component_scores"] = scores
        
        print(f"\n   📊 Data Sufficiency: {scores['data_sufficiency']:.1f}%")
        print(f"   ⚙️ Processing Effectiveness: {scores['processing_effectiveness']:.1f}%")
        print(f"   🧠 Training Convergence: {scores['training_convergence']:.1f}%")
        print(f"   🌉 Bridge Discovery: {scores['bridge_discovery']:.1f}%")
        print(f"   🎯 Overall Score: {scores['overall_score']:.1f}%")
        
        return scores
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete pipeline validation."""
        print("=" * 80)
        print("SEQUENTIAL-ISNE FULL PIPELINE VALIDATION")
        print("=" * 80)
        
        # Run all validations
        self.validate_data_density()
        self.validate_processing_pipeline()
        self.validate_model_training()
        self.validate_theory_practice_bridges()
        scores = self.calculate_overall_validation()
        
        # Determine validation status
        self.results["validation_passed"] = scores["overall_score"] >= 70
        
        # Save results
        output_dir = Path("experiments/validation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = output_dir / f"full_validation_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        print(f"\n🎯 Final Score: {scores['overall_score']:.1f}%")
        print(f"✅ Status: {'PASSED' if self.results['validation_passed'] else 'FAILED'}")
        print(f"💾 Results saved to: {output_file}")
        
        # Print concerns if any
        if scores["data_sufficiency"] < 70:
            print("\n⚠️ CONCERN: Data density is low. Consider adding more test data.")
        if scores["training_convergence"] < 70:
            print("\n⚠️ CONCERN: Model convergence is poor. May need more data or tuning.")
        if scores["bridge_discovery"] < 50:
            print("\n⚠️ CONCERN: Few theory-practice bridges found. Check PDF extraction.")
        
        return self.results


def main():
    """Run the full pipeline validation."""
    validator = ComprehensivePipelineValidator()
    results = validator.run_full_validation()
    
    # Return exit code based on validation
    return 0 if results["validation_passed"] else 1


if __name__ == "__main__":
    exit(main())