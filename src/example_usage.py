"""
Example usage of the document processing pipeline for ISNE testing.

This script demonstrates how to process PDFs and generate embeddings
suitable for Sequential-ISNE experiments.
"""

import logging
from pathlib import Path
from pipeline import DocumentProcessingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main example function."""
    logger = logging.getLogger(__name__)
    
    # Initialize the pipeline with configuration
    config = {
        'document_processor': {
            # Docling configuration
        },
        'chunker': {
            'chunk_size': 512,
            'chunk_overlap': 50,
            'preserve_sentence_boundaries': True
        },
        'embedder': {
            'model_name': 'all-MiniLM-L6-v2',
            'batch_size': 32,
            'normalize_embeddings': True
        }
    }
    
    pipeline = DocumentProcessingPipeline(config)
    
    # Check pipeline health
    health = pipeline.check_health()
    logger.info(f"Pipeline health: {health}")
    
    # Example 1: Process a single PDF file
    pdf_path = Path("../isne-enhanced/Unsupervised Graph Representation Learning with Inductive Shallow Node Embedding.pdf")
    
    if pdf_path.exists():
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Standard processing
        result = pipeline.process_file(
            file_path=pdf_path,
            chunk_size=512,
            chunk_overlap=50,
            model_name='all-MiniLM-L6-v2'
        )
        
        if result["success"]:
            logger.info(f"Successfully processed file:")
            logger.info(f"  - Document ID: {result['document']['id']}")
            logger.info(f"  - Content length: {result['document']['content_length']} characters")
            logger.info(f"  - Number of chunks: {len(result['chunks'])}")
            logger.info(f"  - Number of embeddings: {len(result['embeddings'])}")
            logger.info(f"  - Processing time: {result['processing_stats']['total_processing_time']:.2f} seconds")
            
            # Show sample chunk
            if result['chunks']:
                sample_chunk = result['chunks'][0]
                logger.info(f"  - Sample chunk text (first 100 chars): {sample_chunk['text'][:100]}...")
        else:
            logger.error(f"Failed to process file: {result['error']}")
        
        # Example 2: Extract embeddings for ISNE
        logger.info("\nExtracting embeddings for ISNE...")
        isne_data = pipeline.extract_embeddings_for_isne(
            file_path=pdf_path,
            chunk_size=256,  # Smaller chunks for ISNE
            chunk_overlap=25,
            model_name='all-MiniLM-L6-v2'
        )
        
        if isne_data:
            logger.info(f"ISNE data extracted:")
            logger.info(f"  - Number of nodes: {isne_data['num_nodes']}")
            logger.info(f"  - Embedding dimension: {isne_data['embedding_dimension']}")
            logger.info(f"  - Model name: {isne_data['model_name']}")
            logger.info(f"  - Adjacency matrix shape: {len(isne_data['adjacency_matrix'])}x{len(isne_data['adjacency_matrix'][0]) if isne_data['adjacency_matrix'] else 0}")
            
            # Save embeddings to numpy format for ISNE
            import numpy as np
            
            embeddings_array = np.array(isne_data['embeddings'])
            adjacency_array = np.array(isne_data['adjacency_matrix'])
            
            # Save to files
            output_dir = Path("../output")
            output_dir.mkdir(exist_ok=True)
            
            np.save(output_dir / "embeddings.npy", embeddings_array)
            np.save(output_dir / "adjacency.npy", adjacency_array)
            
            # Save metadata
            import json
            metadata = {
                "source_file": isne_data["source_file"],
                "chunk_ids": isne_data["chunk_ids"],
                "texts": isne_data["texts"],
                "embedding_dimension": isne_data["embedding_dimension"],
                "model_name": isne_data["model_name"],
                "num_nodes": isne_data["num_nodes"],
                "processing_stats": isne_data["processing_stats"]
            }
            
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved ISNE data to {output_dir}/")
        else:
            logger.error("Failed to extract ISNE data")
    
    else:
        logger.warning(f"PDF file not found: {pdf_path}")
        logger.info("You can test with any PDF file by updating the pdf_path variable")
    
    # Get final pipeline statistics
    stats = pipeline.get_pipeline_stats()
    logger.info(f"\nPipeline statistics:")
    logger.info(f"  - Total files processed: {stats['total_files_processed']}")
    logger.info(f"  - Successful files: {stats['successful_files']}")
    logger.info(f"  - Failed files: {stats['failed_files']}")
    logger.info(f"  - Success rate: {stats['success_rate']:.2%}")
    logger.info(f"  - Total chunks created: {stats['total_chunks_created']}")
    logger.info(f"  - Total embeddings created: {stats['total_embeddings_created']}")
    logger.info(f"  - Average processing time: {stats['avg_processing_time']:.2f} seconds")


if __name__ == "__main__":
    main()