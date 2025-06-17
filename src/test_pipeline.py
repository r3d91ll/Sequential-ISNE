"""
Test script for the document processing pipeline.

This script tests the pipeline components with simple text input to verify
functionality without requiring external dependencies.
"""

import logging
from pathlib import Path
from doc_types import ChunkingInput, EmbeddingInput, TextChunk
from chunker import TextChunker
from embedder import TextEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_chunker():
    """Test the text chunker with sample text."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing text chunker...")
    
    # Sample text
    sample_text = """
    This is a test document for the Sequential-ISNE project. 
    We are testing the document processing pipeline components.
    The pipeline includes document processing, text chunking, and embedding generation.
    Each component is designed to work independently and can be configured separately.
    This approach provides flexibility for different use cases and experimental setups.
    The chunker splits text into manageable pieces while preserving semantic boundaries.
    Sentence boundaries are respected to maintain coherent chunk content.
    Overlap between chunks ensures continuity of context across chunk boundaries.
    """
    
    # Initialize chunker
    chunker = TextChunker({'chunk_size': 200, 'chunk_overlap': 50})
    
    # Create chunking input
    chunking_input = ChunkingInput(
        text=sample_text.strip(),
        document_id="test_doc",
        chunk_size=200,
        chunk_overlap=50
    )
    
    # Perform chunking
    result = chunker.chunk(chunking_input)
    
    if result.errors:
        logger.error(f"Chunking failed: {result.errors}")
        return False
    
    logger.info(f"Chunking successful:")
    logger.info(f"  - Total chunks: {len(result.chunks)}")
    logger.info(f"  - Processing time: {result.processing_stats.get('processing_time', 0):.3f} seconds")
    
    for i, chunk in enumerate(result.chunks[:3]):  # Show first 3 chunks
        logger.info(f"  - Chunk {i}: {chunk.text[:100]}...")
    
    return result.chunks

def test_embedder(chunks):
    """Test the text embedder with sample chunks."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing text embedder...")
    
    # Initialize embedder
    embedder = TextEmbedder({'model_name': 'all-MiniLM-L6-v2'})
    
    # Create embedding input
    embedding_input = EmbeddingInput(
        chunks=chunks,
        model_name='all-MiniLM-L6-v2',
        metadata={'test': True}
    )
    
    # Generate embeddings
    result = embedder.embed(embedding_input)
    
    if result.errors:
        logger.error(f"Embedding failed: {result.errors}")
        return False
    
    logger.info(f"Embedding successful:")
    logger.info(f"  - Total embeddings: {len(result.embeddings)}")
    logger.info(f"  - Embedding dimension: {result.model_info.get('embedding_dimension', 0)}")
    logger.info(f"  - Model name: {result.model_info.get('model_name', 'unknown')}")
    logger.info(f"  - Processing time: {result.embedding_stats.get('processing_time', 0):.3f} seconds")
    
    return result.embeddings

def test_isne_format(chunks, embeddings):
    """Test creating ISNE-compatible output format."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing ISNE format creation...")
    
    # Create adjacency matrix (consecutive chunks connected)
    adjacency_matrix = []
    for i in range(len(chunks)):
        row = [0] * len(chunks)
        # Connect to previous and next chunks
        if i > 0:
            row[i-1] = 1
        if i < len(chunks) - 1:
            row[i+1] = 1
        adjacency_matrix.append(row)
    
    # Extract embedding vectors
    embedding_vectors = [emb.embedding for emb in embeddings]
    
    isne_data = {
        "embeddings": embedding_vectors,
        "texts": [chunk.text for chunk in chunks],
        "chunk_ids": [chunk.id for chunk in chunks],
        "adjacency_matrix": adjacency_matrix,
        "embedding_dimension": embeddings[0].embedding_dimension if embeddings else 0,
        "num_nodes": len(chunks)
    }
    
    logger.info(f"ISNE format created:")
    logger.info(f"  - Number of nodes: {isne_data['num_nodes']}")
    logger.info(f"  - Embedding dimension: {isne_data['embedding_dimension']}")
    logger.info(f"  - Adjacency matrix shape: {len(adjacency_matrix)}x{len(adjacency_matrix[0]) if adjacency_matrix else 0}")
    
    # Check adjacency matrix properties
    total_edges = sum(sum(row) for row in adjacency_matrix)
    logger.info(f"  - Total edges: {total_edges}")
    
    return isne_data

def main():
    """Main test function."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting pipeline component tests...")
    
    # Test chunker
    chunks = test_chunker()
    if not chunks:
        logger.error("Chunker test failed")
        return
    
    # Test embedder
    embeddings = test_embedder(chunks)
    if not embeddings:
        logger.error("Embedder test failed")
        return
    
    # Test ISNE format
    isne_data = test_isne_format(chunks, embeddings)
    
    # Save test output
    try:
        import json
        import numpy as np
        
        output_dir = Path("../test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save as JSON for inspection
        test_output = {
            "chunks": [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ],
            "embeddings": [
                {
                    "chunk_id": emb.chunk_id,
                    "embedding": emb.embedding,
                    "dimension": emb.embedding_dimension,
                    "model_name": emb.model_name
                }
                for emb in embeddings
            ],
            "isne_data": isne_data
        }
        
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(test_output, f, indent=2)
        
        # Save numpy arrays for ISNE
        np.save(output_dir / "test_embeddings.npy", np.array(isne_data["embeddings"]))
        np.save(output_dir / "test_adjacency.npy", np.array(isne_data["adjacency_matrix"]))
        
        logger.info(f"Test results saved to {output_dir}/")
        
    except Exception as e:
        logger.warning(f"Could not save test results: {e}")
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main()