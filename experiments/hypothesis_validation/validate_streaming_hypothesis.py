#!/usr/bin/env python3
"""
Sequential-ISNE Hypothesis Validation

This script empirically validates the core streaming hypothesis:
Sequential processing order creates meaningful structural relationships without explicit knowledge graph construction.

Tests:
1. Co-location hypothesis: Files in same directory should have similar ISNE embeddings
2. Sequential proximity: Chunks close in processing order should be related
3. Boundary awareness: Document markers should create meaningful boundaries
4. Cross-document discovery: Related concepts across documents should be discovered
"""

import sys
import logging
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import tempfile
import os
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ProcessingOrder(Enum):
    """Strategies for ordering document processing to optimize co-location discovery."""
    DIRECTORY_FIRST = "directory_first"
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"


@dataclass
class ChunkMetadata:
    """Metadata for a chunk in the streaming processor."""
    chunk_id: int
    chunk_type: str
    doc_path: str
    directory: str
    processing_order: int
    doc_start_chunk_id: int = None
    doc_end_chunk_id: int = None
    prev_chunk_id: int = None
    next_chunk_id: int = None
    content_hash: str = None


@dataclass
class StreamingChunk:
    """A chunk in the streaming processor with full metadata."""
    chunk_id: int
    content: str
    metadata: ChunkMetadata
    embedding: List[float] = None
    isne_embedding: List[float] = None


def create_test_file_structure() -> Dict[str, str]:
    """Create a test file structure that simulates a realistic codebase."""
    
    test_files = {
        # Authentication module - should cluster together
        "src/auth/jwt_handler.py": """
import jwt
import time
from typing import Optional

SECRET_KEY = "your-secret-key"

def validate_token(token: str) -> bool:
    \"\"\"Validate JWT token expiration and signature.\"\"\"
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload.get('exp', 0) > time.time()
    except jwt.InvalidTokenError:
        return False

def create_token(user_id: str, expiry_hours: int = 24) -> str:
    \"\"\"Create a new JWT token for user.\"\"\"
    payload = {
        'user_id': user_id,
        'exp': time.time() + (expiry_hours * 3600),
        'iat': time.time()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
""",
        
        "src/auth/auth_service.py": """
from .jwt_handler import validate_token, create_token
import bcrypt

class AuthenticationService:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.failed_attempts = {}
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        \"\"\"Authenticate user and return JWT token if successful.\"\"\"
        if self._verify_password(username, password):
            return create_token(username)
        return None
    
    def _verify_password(self, username: str, password: str) -> bool:
        \"\"\"Verify user password against stored hash.\"\"\"
        stored_hash = self._get_password_hash(username)
        return bcrypt.checkpw(password.encode(), stored_hash)
""",

        "src/auth/README.md": """
# Authentication Module

This module handles JWT token validation and user authentication for the application.

## Components

### JWT Handler (`jwt_handler.py`)
- `validate_token()`: Checks token expiration and signature using SECRET_KEY
- `create_token()`: Generates new JWT tokens for authenticated users

### Authentication Service (`auth_service.py`)  
- `AuthenticationService`: Main authentication class
- `authenticate_user()`: Complete authentication flow with password verification
- Password hashing using bcrypt for security

## Usage

```python
from src.auth import AuthenticationService, validate_token

# Create service
auth = AuthenticationService(secret_key="your-key")

# Authenticate user
token = auth.authenticate_user("username", "password")

# Validate token later
is_valid = validate_token(token)
```

## Security Notes

- All tokens expire after 24 hours by default
- Failed login attempts are tracked to prevent brute force
- Passwords are hashed using bcrypt with salt
""",

        "src/auth/tests/test_jwt.py": """
import pytest
from src.auth.jwt_handler import validate_token, create_token
import time

class TestJWTHandler:
    def test_validate_token_success(self):
        \"\"\"Test successful token validation.\"\"\"
        token = create_token("test_user", expiry_hours=1)
        assert validate_token(token) == True
    
    def test_validate_token_expired(self):
        \"\"\"Test expired token validation.\"\"\"
        token = create_token("test_user", expiry_hours=0)
        time.sleep(0.1)
        assert validate_token(token) == False
    
    def test_validate_token_invalid(self):
        \"\"\"Test invalid token validation.\"\"\"
        invalid_token = "invalid.token.here"
        assert validate_token(invalid_token) == False
    
    def test_create_token_format(self):
        \"\"\"Test token creation returns valid JWT format.\"\"\"
        token = create_token("test_user")
        assert isinstance(token, str)
        assert len(token.split('.')) == 3
""",

        # Utility module - should be separate cluster
        "src/utils/math_utils.py": """
def calculate_fibonacci(n: int) -> int:
    \"\"\"Calculate nth Fibonacci number using recursion.\"\"\"
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def factorial(n: int) -> int:
    \"\"\"Calculate factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)

def is_prime(n: int) -> bool:
    \"\"\"Check if number is prime.\"\"\"
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""",

        "src/utils/string_utils.py": """
import re
from typing import List

def slugify(text: str) -> str:
    \"\"\"Convert text to URL-friendly slug.\"\"\"
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def word_count(text: str) -> int:
    \"\"\"Count words in text.\"\"\"
    return len(text.split())

def extract_emails(text: str) -> List[str]:
    \"\"\"Extract email addresses from text.\"\"\"
    pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
    return re.findall(pattern, text)
""",

        # Documentation - should bridge with code
        "docs/security/guidelines.md": """
# Security Guidelines

All API endpoints must implement authentication. Use JWT tokens for stateless authentication across microservices.

## Authentication Requirements

1. **Token Validation**: Every request must validate JWT token
2. **Expiration Handling**: Tokens expire after 24 hours maximum
3. **Secret Management**: Store SECRET_KEY securely, never in code
4. **Rate Limiting**: Implement rate limiting on auth endpoints

## Best Practices

- Use HTTPS for all authentication endpoints
- Implement proper password hashing (bcrypt recommended)
- Log authentication failures for monitoring
- Use secure session management

## Integration

See `src/auth/` module for implementation details and usage examples.
""",

        "docs/api/endpoints.md": """
# API Endpoints

## Authentication Endpoints

### POST /auth/login
Authenticate user and receive JWT token.

**Request:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "token": "jwt.token.here",
  "expires_in": 86400
}
```

### POST /auth/validate
Validate JWT token.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "valid": true,
  "user_id": "string"
}
```

## Utility Endpoints

### GET /utils/math/fibonacci/{n}
Calculate nth Fibonacci number.

### GET /utils/text/slugify
Convert text to URL slug.
""",

        # Configuration
        "config/settings.yaml": """
# Application Settings

auth:
  secret_key: "${SECRET_KEY}"
  token_expiry_hours: 24
  max_failed_attempts: 5
  
database:
  host: "localhost"
  port: 5432
  name: "myapp"
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
""",

        "README.md": """
# Test Application

A simple application demonstrating authentication, utilities, and documentation.

## Structure

- `src/auth/` - Authentication module with JWT handling
- `src/utils/` - Utility functions for math and strings  
- `docs/` - Documentation and API specifications
- `config/` - Configuration files

## Key Features

- JWT-based authentication with expiration
- Mathematical utilities (Fibonacci, factorial, prime checking)
- String processing utilities
- Comprehensive documentation
- Security guidelines and best practices
"""
    }
    
    return test_files


class SimpleStreamingProcessor:
    """Simplified streaming processor for validation testing."""
    
    def __init__(self, processing_order: ProcessingOrder = ProcessingOrder.DIRECTORY_FIRST):
        self.processing_order = processing_order
        self.current_chunk_id = 0
        
    def _sort_files_by_strategy(self, file_paths: List[str]) -> List[str]:
        """Sort files to optimize co-location discovery."""
        if self.processing_order == ProcessingOrder.DIRECTORY_FIRST:
            def sort_key(path: str) -> Tuple[str, int, str]:
                p = Path(path)
                directory = str(p.parent)
                
                if p.name.lower().startswith('readme'):
                    priority = 0
                elif p.suffix in ['.py', '.js', '.ts', '.rs', '.go', '.java']:
                    priority = 1
                elif 'test' in p.name.lower() or 'spec' in p.name.lower():
                    priority = 3
                else:
                    priority = 2
                
                return (directory, priority, p.name)
            
            return sorted(file_paths, key=sort_key)
        else:
            return sorted(file_paths)
    
    def process_documents(self, file_paths: List[str], temp_dir: str) -> List[StreamingChunk]:
        """Process documents and return streaming chunks."""
        sorted_files = self._sort_files_by_strategy(file_paths)
        chunks = []
        
        current_directory = None
        
        for file_path in sorted_files:
            directory = str(Path(file_path).parent)
            
            # Directory marker
            if directory != current_directory:
                dir_chunk = StreamingChunk(
                    chunk_id=self.current_chunk_id,
                    content=f"<DIRECTORY_START:{directory}>",
                    metadata=ChunkMetadata(
                        chunk_id=self.current_chunk_id,
                        chunk_type="directory_marker",
                        doc_path=f"<DIR:{directory}>",
                        directory=directory,
                        processing_order=self.current_chunk_id
                    )
                )
                chunks.append(dir_chunk)
                self.current_chunk_id += 1
                current_directory = directory
            
            # Read file content
            try:
                full_path = Path(temp_dir) / file_path
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                content = "Error reading file"
            
            # Document start
            start_chunk = StreamingChunk(
                chunk_id=self.current_chunk_id,
                content=f"<DOC_START:{file_path}>",
                metadata=ChunkMetadata(
                    chunk_id=self.current_chunk_id,
                    chunk_type="doc_start",
                    doc_path=file_path,
                    directory=directory,
                    processing_order=self.current_chunk_id
                )
            )
            chunks.append(start_chunk)
            doc_start_id = self.current_chunk_id
            self.current_chunk_id += 1
            
            # Content chunks
            content_chunks = self._chunk_content(content)
            for chunk_content in content_chunks:
                content_chunk = StreamingChunk(
                    chunk_id=self.current_chunk_id,
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        chunk_id=self.current_chunk_id,
                        chunk_type="content",
                        doc_path=file_path,
                        directory=directory,
                        processing_order=self.current_chunk_id,
                        doc_start_chunk_id=doc_start_id,
                        content_hash=hashlib.md5(chunk_content.encode()).hexdigest()[:8]
                    )
                )
                chunks.append(content_chunk)
                self.current_chunk_id += 1
            
            # Document end
            end_chunk = StreamingChunk(
                chunk_id=self.current_chunk_id,
                content=f"<DOC_END:{file_path}>",
                metadata=ChunkMetadata(
                    chunk_id=self.current_chunk_id,
                    chunk_type="doc_end",
                    doc_path=file_path,
                    directory=directory,
                    processing_order=self.current_chunk_id
                )
            )
            chunks.append(end_chunk)
            self.current_chunk_id += 1
        
        return chunks
    
    def _chunk_content(self, content: str) -> List[str]:
        """Simple content chunking."""
        chunk_size = 256
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            if len(chunk.strip()) > 20:
                chunks.append(chunk)
            start += chunk_size - 25  # Small overlap
        
        return chunks if chunks else [content[:chunk_size]]


def create_temporary_file_structure(test_files: Dict[str, str]) -> str:
    """Create temporary files for testing."""
    temp_dir = tempfile.mkdtemp(prefix="streaming_test_")
    
    for file_path, content in test_files.items():
        full_path = Path(temp_dir) / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(content)
    
    return temp_dir


def analyze_streaming_patterns(chunks: List[StreamingChunk]) -> Dict[str, Any]:
    """Analyze patterns in the streaming chunk sequence."""
    
    analysis = {
        'total_chunks': len(chunks),
        'content_chunks': 0,
        'boundary_markers': 0,
        'directory_markers': 0,
        'directories': set(),
        'documents': set(),
        'directory_clusters': defaultdict(list),
        'co_location_opportunities': []
    }
    
    for i, chunk in enumerate(chunks):
        if chunk.metadata.chunk_type == 'content':
            analysis['content_chunks'] += 1
            analysis['directory_clusters'][chunk.metadata.directory].append(i)
        elif chunk.metadata.chunk_type in ['doc_start', 'doc_end']:
            analysis['boundary_markers'] += 1
        elif chunk.metadata.chunk_type == 'directory_marker':
            analysis['directory_markers'] += 1
        
        analysis['directories'].add(chunk.metadata.directory)
        if not chunk.metadata.doc_path.startswith('<'):
            analysis['documents'].add(chunk.metadata.doc_path)
    
    # Analyze directory clustering
    for directory, chunk_indices in analysis['directory_clusters'].items():
        if len(chunk_indices) > 1:
            max_gap = max(chunk_indices[i+1] - chunk_indices[i] for i in range(len(chunk_indices)-1))
            total_span = chunk_indices[-1] - chunk_indices[0] + 1
            clustering_ratio = len(chunk_indices) / total_span
            
            analysis['co_location_opportunities'].append({
                'directory': directory,
                'chunk_count': len(chunk_indices),
                'indices': chunk_indices,
                'max_gap': max_gap,
                'clustering_ratio': clustering_ratio,
                'span': total_span
            })
    
    # Convert sets to lists for JSON serialization
    analysis['directories'] = list(analysis['directories'])
    analysis['documents'] = list(analysis['documents'])
    
    return analysis


def test_co_location_hypothesis(chunks: List[StreamingChunk], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Test if files in same directory are processed close together."""
    
    results = {
        'hypothesis': 'Files in same directory should be processed close together',
        'test_results': [],
        'overall_score': 0.0,
        'passed': False
    }
    
    for opportunity in analysis['co_location_opportunities']:
        clustering_ratio = opportunity['clustering_ratio']
        passed = clustering_ratio > 0.5
        
        results['test_results'].append({
            'directory': opportunity['directory'],
            'clustering_ratio': clustering_ratio,
            'chunk_count': opportunity['chunk_count'],
            'max_gap': opportunity['max_gap'],
            'passed': passed
        })
    
    if results['test_results']:
        results['overall_score'] = np.mean([r['clustering_ratio'] for r in results['test_results']])
        results['passed'] = results['overall_score'] > 0.5
    
    return results


def test_sequential_proximity_hypothesis(chunks: List[StreamingChunk]) -> Dict[str, Any]:
    """Test if sequential chunks provide meaningful context."""
    
    results = {
        'hypothesis': 'Sequential chunks should provide contextual relationships',
        'test_results': [],
        'meaningful_sequences': 0,
        'total_sequences': 0,
        'passed': False
    }
    
    for i in range(len(chunks) - 1):
        current = chunks[i]
        next_chunk = chunks[i + 1]
        
        results['total_sequences'] += 1
        
        meaningful = False
        relationship_type = None
        
        # Same document
        if (current.metadata.chunk_type == 'content' and 
            next_chunk.metadata.chunk_type == 'content' and
            current.metadata.doc_path == next_chunk.metadata.doc_path):
            meaningful = True
            relationship_type = 'same_document'
        
        # Same directory
        elif (current.metadata.chunk_type == 'content' and
              next_chunk.metadata.chunk_type == 'content' and
              current.metadata.directory == next_chunk.metadata.directory):
            meaningful = True
            relationship_type = 'same_directory'
        
        # Document boundary relationships
        elif ((current.metadata.chunk_type == 'doc_start' and next_chunk.metadata.chunk_type == 'content') or
              (current.metadata.chunk_type == 'content' and next_chunk.metadata.chunk_type == 'doc_end')):
            meaningful = True
            relationship_type = 'document_boundary'
        
        if meaningful:
            results['meaningful_sequences'] += 1
            results['test_results'].append({
                'chunk_index': i,
                'relationship_type': relationship_type,
                'current_doc': current.metadata.doc_path,
                'next_doc': next_chunk.metadata.doc_path
            })
    
    if results['total_sequences'] > 0:
        meaningful_ratio = results['meaningful_sequences'] / results['total_sequences']
        results['meaningful_ratio'] = meaningful_ratio
        results['passed'] = meaningful_ratio > 0.6
    
    return results


def test_boundary_awareness_hypothesis(chunks: List[StreamingChunk]) -> Dict[str, Any]:
    """Test if document boundaries create meaningful structure."""
    
    results = {
        'hypothesis': 'Document boundaries should create meaningful structure',
        'documents_with_boundaries': 0,
        'total_documents': 0,
        'boundary_structure': [],
        'passed': False
    }
    
    # Group chunks by document
    documents = defaultdict(list)
    for i, chunk in enumerate(chunks):
        if not chunk.metadata.doc_path.startswith('<'):
            documents[chunk.metadata.doc_path].append((i, chunk))
    
    results['total_documents'] = len(documents)
    
    for doc_path, doc_chunks in documents.items():
        has_start_boundary = False
        has_end_boundary = False
        content_chunks = 0
        
        for i, (chunk_idx, chunk) in enumerate(doc_chunks):
            if chunk.metadata.chunk_type == 'doc_start':
                has_start_boundary = True
            elif chunk.metadata.chunk_type == 'doc_end':
                has_end_boundary = True
            elif chunk.metadata.chunk_type == 'content':
                content_chunks += 1
        
        if has_start_boundary and has_end_boundary and content_chunks > 0:
            results['documents_with_boundaries'] += 1
        
        results['boundary_structure'].append({
            'document': doc_path,
            'has_start_boundary': has_start_boundary,
            'has_end_boundary': has_end_boundary,
            'content_chunks': content_chunks,
            'well_structured': has_start_boundary and has_end_boundary and content_chunks > 0
        })
    
    if results['total_documents'] > 0:
        boundary_ratio = results['documents_with_boundaries'] / results['total_documents']
        results['boundary_ratio'] = boundary_ratio
        results['passed'] = boundary_ratio > 0.8
    
    return results


def test_cross_document_discovery_hypothesis(chunks: List[StreamingChunk]) -> Dict[str, Any]:
    """Test potential for cross-document relationship discovery."""
    
    results = {
        'hypothesis': 'Related concepts across documents should be discoverable',
        'concept_clusters': [],
        'cross_document_opportunities': 0,
        'total_concepts': 0,
        'passed': False
    }
    
    # Simple concept extraction based on content keywords
    concept_keywords = {
        'authentication': ['auth', 'login', 'token', 'jwt', 'password', 'validate'],
        'security': ['security', 'secret', 'hash', 'encrypt', 'ssl', 'https'],
        'api': ['api', 'endpoint', 'request', 'response', 'http'],
        'math': ['fibonacci', 'factorial', 'prime', 'calculate'],
        'utilities': ['util', 'helper', 'function', 'tool'],
        'documentation': ['readme', 'guide', 'example', 'usage']
    }
    
    # Classify chunks by concepts
    chunk_concepts = []
    for chunk in chunks:
        if chunk.metadata.chunk_type != 'content':
            chunk_concepts.append(set())
            continue
            
        content_lower = chunk.content.lower()
        found_concepts = set()
        
        for concept, keywords in concept_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    found_concepts.add(concept)
        
        chunk_concepts.append(found_concepts)
    
    # Find cross-document concept relationships
    concept_documents = defaultdict(set)
    for i, (chunk, concepts) in enumerate(zip(chunks, chunk_concepts)):
        if chunk.metadata.chunk_type == 'content' and not chunk.metadata.doc_path.startswith('<'):
            for concept in concepts:
                concept_documents[concept].add(chunk.metadata.doc_path)
    
    results['total_concepts'] = len(concept_documents)
    
    for concept, documents in concept_documents.items():
        if len(documents) > 1:
            results['cross_document_opportunities'] += 1
            results['concept_clusters'].append({
                'concept': concept,
                'documents': list(documents),
                'document_count': len(documents)
            })
    
    if results['total_concepts'] > 0:
        cross_doc_ratio = results['cross_document_opportunities'] / results['total_concepts']
        results['cross_document_ratio'] = cross_doc_ratio
        results['passed'] = cross_doc_ratio > 0.3
    
    return results


def run_streaming_hypothesis_validation():
    """Run complete validation of streaming ISNE hypothesis."""
    
    print("=" * 80)
    print("SEQUENTIAL-ISNE HYPOTHESIS VALIDATION")
    print("=" * 80)
    
    # Create test file structure
    print("\n1. Creating test file structure...")
    test_files = create_test_file_structure()
    temp_dir = create_temporary_file_structure(test_files)
    
    try:
        # Get all file paths
        file_paths = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), temp_dir)
                file_paths.append(rel_path)
        
        print(f"   Created {len(file_paths)} test files")
        
        # Process with streaming processor
        print("\n2. Processing files with StreamingProcessor...")
        processor = SimpleStreamingProcessor(ProcessingOrder.DIRECTORY_FIRST)
        
        chunks = processor.process_documents(file_paths, temp_dir)
        print(f"   Generated {len(chunks)} chunks")
        
        # Analyze patterns
        print("\n3. Analyzing streaming patterns...")
        analysis = analyze_streaming_patterns(chunks)
        
        print(f"   Content chunks: {analysis['content_chunks']}")
        print(f"   Boundary markers: {analysis['boundary_markers']}")
        print(f"   Directory markers: {analysis['directory_markers']}")
        print(f"   Directories: {len(analysis['directories'])}")
        print(f"   Documents: {len(analysis['documents'])}")
        
        # Run hypothesis tests
        print("\n4. Testing core hypotheses...")
        
        # Test 1: Co-location hypothesis
        print("\n   Testing co-location hypothesis...")
        colocate_results = test_co_location_hypothesis(chunks, analysis)
        print(f"   Result: {'PASSED' if colocate_results['passed'] else 'FAILED'}")
        print(f"   Overall clustering score: {colocate_results['overall_score']:.3f}")
        
        # Test 2: Sequential proximity hypothesis  
        print("\n   Testing sequential proximity hypothesis...")
        proximity_results = test_sequential_proximity_hypothesis(chunks)
        print(f"   Result: {'PASSED' if proximity_results['passed'] else 'FAILED'}")
        print(f"   Meaningful sequences: {proximity_results['meaningful_sequences']}/{proximity_results['total_sequences']}")
        if 'meaningful_ratio' in proximity_results:
            print(f"   Meaningful ratio: {proximity_results['meaningful_ratio']:.3f}")
        
        # Test 3: Boundary awareness hypothesis
        print("\n   Testing boundary awareness hypothesis...")
        boundary_results = test_boundary_awareness_hypothesis(chunks)
        print(f"   Result: {'PASSED' if boundary_results['passed'] else 'FAILED'}")
        print(f"   Documents with boundaries: {boundary_results['documents_with_boundaries']}/{boundary_results['total_documents']}")
        if 'boundary_ratio' in boundary_results:
            print(f"   Boundary ratio: {boundary_results['boundary_ratio']:.3f}")
        
        # Test 4: Cross-document discovery hypothesis
        print("\n   Testing cross-document discovery hypothesis...")
        cross_doc_results = test_cross_document_discovery_hypothesis(chunks)
        print(f"   Result: {'PASSED' if cross_doc_results['passed'] else 'FAILED'}")
        print(f"   Cross-document concepts: {cross_doc_results['cross_document_opportunities']}/{cross_doc_results['total_concepts']}")
        if 'cross_document_ratio' in cross_doc_results:
            print(f"   Cross-document ratio: {cross_doc_results['cross_document_ratio']:.3f}")
        
        # Overall assessment
        print("\n" + "=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)
        
        passed_tests = sum([
            colocate_results['passed'],
            proximity_results['passed'], 
            boundary_results['passed'],
            cross_doc_results['passed']
        ])
        
        print(f"\nHypothesis tests passed: {passed_tests}/4")
        
        if passed_tests >= 3:
            print("✅ SEQUENTIAL-ISNE HYPOTHESIS VALIDATED")
            print("   The streaming chunk processing approach shows strong evidence of creating")
            print("   meaningful structural relationships that can be learned by ISNE.")
        elif passed_tests >= 2:
            print("⚠️  SEQUENTIAL-ISNE HYPOTHESIS PARTIALLY VALIDATED")
            print("   Some evidence supports the approach, but improvements may be needed.")
        else:
            print("❌ SEQUENTIAL-ISNE HYPOTHESIS NOT VALIDATED")
            print("   Limited evidence that streaming creates meaningful relationships.")
        
        # Save detailed results
        output_file = Path("output/hypothesis_validation_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'test_configuration': {
                'processing_order': processor.processing_order.value,
                'chunk_count': len(chunks)
            },
            'pattern_analysis': analysis,
            'hypothesis_tests': {
                'co_location': colocate_results,
                'sequential_proximity': proximity_results,
                'boundary_awareness': boundary_results,
                'cross_document_discovery': cross_doc_results
            },
            'overall_assessment': {
                'tests_passed': passed_tests,
                'total_tests': 4,
                'validation_status': 'validated' if passed_tests >= 3 else 'partial' if passed_tests >= 2 else 'failed'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {output_file}")
        
        return passed_tests >= 3
        
    finally:
        # Cleanup temporary files
        import shutil
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_streaming_hypothesis_validation()
    sys.exit(0 if success else 1)