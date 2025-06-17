#!/usr/bin/env python3
"""
Test script for data normalization pipeline.

Tests the file processor and data normalizer on our test data to ensure
proper file type routing and JSON normalization.
"""

import json
import logging
from pathlib import Path
from src.data_normalizer import DataNormalizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_normalization():
    """Test the normalization pipeline on test data."""
    
    print("🔄 Testing Data Normalization Pipeline")
    print("=" * 60)
    
    normalizer = DataNormalizer()
    
    # Test directory
    test_dir = Path("/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata/isne-testdata")
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print(f"📁 Processing directory: {test_dir}")
    
    # Normalize all files in the directory
    normalized_docs = normalizer.normalize_directory(test_dir, recursive=True)
    
    print(f"\n📊 Processing Results:")
    print(f"   Total documents processed: {len(normalized_docs)}")
    
    # Analyze results by file type
    file_types = {}
    successful = 0
    failed = 0
    
    for doc in normalized_docs:
        file_type = doc['source']['file_type']
        file_types[file_type] = file_types.get(file_type, 0) + 1
        
        if doc['error']:
            failed += 1
        else:
            successful += 1
    
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"\n📋 File Types Processed:")
    for file_type, count in file_types.items():
        print(f"   {file_type}: {count} files")
    
    # Show sample results
    print(f"\n🔍 Sample Results:")
    
    # Find a Python file
    python_docs = [doc for doc in normalized_docs if doc['source']['file_type'] == 'python' and not doc['error']]
    if python_docs:
        py_doc = python_docs[0]
        print(f"\n📄 Python File: {py_doc['source']['file_name']}")
        print(f"   Functions: {py_doc['content']['content_summary'].get('functions', 0)}")
        print(f"   Classes: {py_doc['content']['content_summary'].get('classes', 0)}")
        print(f"   Chunks: {len(py_doc['chunks'])}")
        
        # Show first chunk
        if py_doc['chunks']:
            chunk = py_doc['chunks'][0]
            print(f"   Sample chunk type: {chunk['chunk_type']}")
            print(f"   Sample content: {chunk['content'][:100]}...")
    
    # Find a document file
    doc_docs = [doc for doc in normalized_docs if doc['source']['file_type'] == 'document' and not doc['error']]
    if doc_docs:
        doc_doc = doc_docs[0]
        print(f"\n📄 Document File: {doc_doc['source']['file_name']}")
        print(f"   Content length: {doc_doc['content']['content_summary'].get('content_length', 0)}")
        print(f"   Chunks: {len(doc_doc['chunks'])}")
        
        # Show first chunk
        if doc_doc['chunks']:
            chunk = doc_doc['chunks'][0]
            print(f"   Sample chunk type: {chunk['chunk_type']}")
            print(f"   Sample content: {chunk['content'][:100]}...")
    
    # Save results for inspection
    output_path = Path("experiments/normalized_data_test.json")
    normalizer.save_normalized_data(normalized_docs, output_path)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Test theory-practice bridge potential
    python_chunks = []
    document_chunks = []
    
    for doc in normalized_docs:
        if not doc['error']:
            if doc['source']['file_type'] == 'python':
                python_chunks.extend(doc['chunks'])
            elif doc['source']['file_type'] == 'document':
                document_chunks.extend(doc['chunks'])
    
    print(f"\n🌉 Theory-Practice Bridge Potential:")
    print(f"   Python chunks: {len(python_chunks)}")
    print(f"   Document chunks: {len(document_chunks)}")
    print(f"   Possible bridges: {len(python_chunks) * len(document_chunks)}")
    
    # Test embedding preparation
    all_chunks = []
    for doc in normalized_docs:
        if not doc['error']:
            all_chunks.extend(doc['chunks'])
    
    print(f"\n🔗 Embedding Preparation:")
    print(f"   Total chunks for embedding: {len(all_chunks)}")
    print(f"   Average chunk size: {sum(len(c['content']) for c in all_chunks) / len(all_chunks) if all_chunks else 0:.1f} chars")
    
    print(f"\n✅ Data Normalization Test Complete!")
    
    return normalized_docs


if __name__ == "__main__":
    test_normalization()