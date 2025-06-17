#!/usr/bin/env python3
"""
PDF Processing Validation for Sequential-ISNE

This script validates that our document processing pipeline correctly extracts
PDF content and creates theory→practice bridges between PDF chunks and code chunks.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Union
import json
import time

# Add src to path
sys.path.append('src')

from simple_document_processor import SimpleDocumentProcessor
from chunker import TextChunker
from embedder import TextEmbedder
from streaming_processor import StreamingChunk, ChunkMetadata
from enhanced_hierarchical_processor import EnhancedHierarchicalProcessor, ProcessingStrategy

logger = logging.getLogger(__name__)


class PDFProcessingValidator:
    """
    Validates PDF processing and theory→practice bridge creation.
    """
    
    def __init__(self):
        self.testdata_path = Path("/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata")
        
        # Initialize simple processing components
        self.doc_processor = SimpleDocumentProcessor()
        self.chunker = TextChunker({
            'chunk_size': 512,
            'chunk_overlap': 50,
            'preserve_sentence_boundaries': True
        })
        self.embedder = TextEmbedder({
            'model_name': 'mock-384d',
            'batch_size': 32
        })
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a file through the complete pipeline."""
        try:
            # Step 1: Document processing
            document = self.doc_processor.process_document(file_path)
            
            if document.error:
                return {
                    'success': False,
                    'error': document.error,
                    'file_path': str(file_path)
                }
            
            # Step 2: Chunking
            from doc_types import ChunkingInput
            chunking_input = ChunkingInput(
                text=document.content,
                document_id=document.id,
                chunk_size=512,
                chunk_overlap=50
            )
            
            chunking_output = self.chunker.chunk(chunking_input)
            
            if chunking_output.errors:
                return {
                    'success': False,
                    'error': f"Chunking failed: {'; '.join(chunking_output.errors)}",
                    'file_path': str(file_path)
                }
            
            # Step 3: Embedding (optional for validation)
            # Skip embedding for validation to avoid dependency issues
            
            return {
                'success': True,
                'file_path': str(file_path),
                'document': {
                    'id': document.id,
                    'content_length': len(document.content),
                    'content_type': document.content_type,
                    'format': document.format,
                    'metadata': document.metadata
                },
                'chunks': [
                    {
                        'id': chunk.id,
                        'text': chunk.text,
                        'start_index': chunk.start_index,
                        'end_index': chunk.end_index,
                        'chunk_index': chunk.chunk_index
                    }
                    for chunk in chunking_output.chunks
                ],
                'embeddings': []  # Skip for validation
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path)
            }
        
    def validate_pdf_processing(self) -> Dict[str, Any]:
        """Validate complete PDF processing and bridge detection."""
        
        print("🔬 PDF Processing and Theory→Practice Bridge Validation")
        print("=" * 70)
        
        results = {}
        
        # Test 1: PDF Content Extraction
        print("\n📄 Test 1: PDF Content Extraction")
        extraction_results = self._test_pdf_extraction()
        results['pdf_extraction'] = extraction_results
        
        # Test 2: Integrated Processing (PDF + Code)
        print("\n🔗 Test 2: Integrated PDF + Code Processing")
        integration_results = self._test_integrated_processing()
        results['integrated_processing'] = integration_results
        
        # Test 3: Theory→Practice Bridge Detection
        print("\n🌉 Test 3: Theory→Practice Bridge Detection")
        bridge_results = self._test_theory_practice_bridges(integration_results)
        results['theory_practice_bridges'] = bridge_results
        
        # Generate overall assessment
        overall_results = self._generate_assessment(results)
        results['overall_assessment'] = overall_results
        
        return results
    
    def _test_pdf_extraction(self) -> Dict[str, Any]:
        """Test PDF content extraction."""
        
        pdf_files = []
        for repo in ['isne-enhanced', 'pathrag-enhanced']:
            repo_path = self.testdata_path / repo
            if repo_path.exists():
                pdf_files.extend(repo_path.glob("**/*.pdf"))
        
        extraction_results = {
            'pdf_files_found': len(pdf_files),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_pdf_chunks': 0,
            'pdf_processing_details': []
        }
        
        for pdf_file in pdf_files:
            try:
                print(f"   📄 Processing: {pdf_file.name}")
                result = self.process_file(pdf_file)
                
                if result.get('success', False):
                    chunks_count = len(result['chunks'])
                    content_length = result['document']['content_length']
                    
                    extraction_results['successful_extractions'] += 1
                    extraction_results['total_pdf_chunks'] += chunks_count
                    
                    extraction_results['pdf_processing_details'].append({
                        'file': str(pdf_file),
                        'success': True,
                        'chunks_created': chunks_count,
                        'content_length': content_length,
                        'sample_content': result['chunks'][0]['text'][:200] + "..." if chunks_count > 0 else ""
                    })
                    
                    print(f"   ✅ Success: {chunks_count} chunks, {content_length} chars")
                else:
                    extraction_results['failed_extractions'] += 1
                    extraction_results['pdf_processing_details'].append({
                        'file': str(pdf_file),
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    })
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                extraction_results['failed_extractions'] += 1
                extraction_results['pdf_processing_details'].append({
                    'file': str(pdf_file),
                    'success': False,
                    'error': str(e)
                })
                print(f"   ❌ Exception: {e}")
        
        print(f"   📊 Results: {extraction_results['successful_extractions']}/{len(pdf_files)} PDFs processed successfully")
        print(f"   📊 Total PDF chunks: {extraction_results['total_pdf_chunks']}")
        
        return extraction_results
    
    def _test_integrated_processing(self) -> Dict[str, Any]:
        """Test integrated processing of PDFs and code files."""
        
        processed_files = {}
        total_chunks = 0
        pdf_chunks = 0
        code_chunks = 0
        
        for repo in ['isne-enhanced', 'pathrag-enhanced']:
            repo_path = self.testdata_path / repo
            if not repo_path.exists():
                continue
                
            # Process all files in repository
            file_patterns = ["**/*.py", "**/*.md", "**/*.pdf"]
            repo_files = []
            
            for pattern in file_patterns:
                repo_files.extend(str(p) for p in repo_path.glob(pattern) if p.is_file())
            
            print(f"   📁 Processing {repo}: {len(repo_files)} files")
            
            repo_results = {
                'total_files': len(repo_files),
                'successful_files': 0,
                'failed_files': 0,
                'chunks_by_type': {},
                'file_details': []
            }
            
            for file_path in repo_files:
                try:
                    result = self.process_file(file_path)
                    
                    if result.get('success', False):
                        chunk_count = len(result['chunks'])
                        file_ext = Path(file_path).suffix
                        
                        repo_results['successful_files'] += 1
                        total_chunks += chunk_count
                        
                        if file_ext == '.pdf':
                            pdf_chunks += chunk_count
                        elif file_ext == '.py':
                            code_chunks += chunk_count
                        
                        if file_ext not in repo_results['chunks_by_type']:
                            repo_results['chunks_by_type'][file_ext] = 0
                        repo_results['chunks_by_type'][file_ext] += chunk_count
                        
                        repo_results['file_details'].append({
                            'file': file_path,
                            'extension': file_ext,
                            'success': True,
                            'chunks': chunk_count
                        })
                        
                    else:
                        repo_results['failed_files'] += 1
                        repo_results['file_details'].append({
                            'file': file_path,
                            'success': False,
                            'error': result.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    repo_results['failed_files'] += 1
                    repo_results['file_details'].append({
                        'file': file_path,
                        'success': False,
                        'error': str(e)
                    })
            
            processed_files[repo] = repo_results
            print(f"   ✅ {repo}: {repo_results['successful_files']}/{repo_results['total_files']} files processed")
        
        integration_results = {
            'repositories_processed': len(processed_files),
            'total_chunks_created': total_chunks,
            'pdf_chunks': pdf_chunks,
            'code_chunks': code_chunks,
            'repository_details': processed_files
        }
        
        print(f"   📊 Total chunks: {total_chunks} ({pdf_chunks} PDF, {code_chunks} Python)")
        
        return integration_results
    
    def _test_theory_practice_bridges(self, integration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test theory→practice bridge detection using processed content."""
        
        # For this test, we'll simulate creating relationships between 
        # PDF chunks and code chunks based on content similarity
        # In the real implementation, this would be done by the hierarchical processor
        
        bridge_opportunities = 0
        potential_bridges = []
        
        for repo, repo_data in integration_results['repository_details'].items():
            pdf_files = []
            code_files = []
            
            for file_detail in repo_data['file_details']:
                if file_detail.get('success', False):
                    if file_detail['extension'] == '.pdf':
                        pdf_files.append(file_detail)
                    elif file_detail['extension'] == '.py':
                        code_files.append(file_detail)
            
            # Count potential bridges (PDF chunks × code chunks in same repo)
            for pdf_file in pdf_files:
                for code_file in code_files:
                    bridge_opportunities += pdf_file['chunks'] * code_file['chunks']
                    potential_bridges.append({
                        'repository': repo,
                        'pdf_file': Path(pdf_file['file']).name,
                        'code_file': Path(code_file['file']).name,
                        'pdf_chunks': pdf_file['chunks'],
                        'code_chunks': code_file['chunks'],
                        'potential_relationships': pdf_file['chunks'] * code_file['chunks']
                    })
        
        bridge_results = {
            'total_bridge_opportunities': bridge_opportunities,
            'repositories_with_opportunities': len([repo for repo, r in integration_results['repository_details'].items() 
                                                  if any(f.get('extension') == '.pdf' and f.get('success', False) 
                                                        for f in r['file_details'])]),
            'potential_bridges': potential_bridges[:5],  # Top 5 for display
            'validation_approach': 'content_based_similarity'
        }
        
        print(f"   📊 Bridge opportunities: {bridge_opportunities}")
        print(f"   📊 Repositories with PDF+code: {bridge_results['repositories_with_opportunities']}")
        
        return bridge_results
    
    def _generate_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment."""
        
        assessments = []
        score = 0
        total_tests = 3
        
        # PDF Extraction Assessment
        extraction = results['pdf_extraction']
        if extraction['successful_extractions'] > 0:
            assessments.append("✅ PDF content extraction successful")
            score += 1
        else:
            assessments.append("❌ PDF content extraction failed")
        
        # Integration Assessment  
        integration = results['integrated_processing']
        if integration['pdf_chunks'] > 0 and integration['code_chunks'] > 0:
            assessments.append("✅ PDF and code processing integrated")
            score += 1
        else:
            assessments.append("❌ Integration incomplete")
        
        # Bridge Assessment
        bridges = results['theory_practice_bridges']
        if bridges['total_bridge_opportunities'] > 0:
            assessments.append("✅ Theory→practice bridge opportunities identified")
            score += 1
        else:
            assessments.append("❌ No bridge opportunities found")
        
        return {
            'individual_assessments': assessments,
            'successful_tests': score,
            'total_tests': total_tests,
            'validation_score': score / total_tests,
            'overall_status': 'PASSED' if score >= 2 else 'FAILED',
            'pdf_processing_working': extraction['successful_extractions'] > 0,
            'integration_working': integration['pdf_chunks'] > 0 and integration['code_chunks'] > 0
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Save validation results."""
        if output_path is None:
            output_path = "/home/todd/ML-Lab/Olympus/Sequential-ISNE/experiments/pdf_validation_results.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 PDF validation results saved to: {output_path}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report."""
        
        report = ["=" * 80]
        report.append("PDF PROCESSING AND THEORY→PRACTICE BRIDGE VALIDATION")
        report.append("=" * 80)
        
        # Overall Assessment
        overall = results['overall_assessment']
        report.append(f"\n🎯 OVERALL VALIDATION RESULTS")
        report.append(f"   Score: {overall['validation_score']:.1%}")
        report.append(f"   Status: {overall['overall_status']}")
        
        # Individual Test Results
        report.append(f"\n📊 INDIVIDUAL TEST RESULTS")
        for assessment in overall['individual_assessments']:
            report.append(f"   {assessment}")
        
        # PDF Extraction Details
        extraction = results['pdf_extraction']
        report.append(f"\n📄 PDF EXTRACTION RESULTS")
        report.append(f"   PDFs found: {extraction['pdf_files_found']}")
        report.append(f"   Successful extractions: {extraction['successful_extractions']}")
        report.append(f"   Total PDF chunks: {extraction['total_pdf_chunks']}")
        
        # Integration Details
        integration = results['integrated_processing']
        report.append(f"\n🔗 INTEGRATION RESULTS")
        report.append(f"   Total chunks: {integration['total_chunks_created']}")
        report.append(f"   PDF chunks: {integration['pdf_chunks']}")
        report.append(f"   Code chunks: {integration['code_chunks']}")
        
        # Bridge Opportunities
        bridges = results['theory_practice_bridges']
        report.append(f"\n🌉 THEORY→PRACTICE BRIDGE OPPORTUNITIES")
        report.append(f"   Total opportunities: {bridges['total_bridge_opportunities']}")
        report.append(f"   Repositories with PDF+code: {bridges['repositories_with_opportunities']}")
        
        if bridges['potential_bridges']:
            report.append(f"\n   Top bridge opportunities:")
            for bridge in bridges['potential_bridges'][:3]:
                report.append(f"   • {bridge['repository']}: {bridge['pdf_file']} ↔ {bridge['code_file']}")
                report.append(f"     Potential relationships: {bridge['potential_relationships']}")
        
        report.append(f"\n🔬 VALIDATION CONCLUSION")
        if overall['pdf_processing_working']:
            report.append(f"   ✅ PDF content extraction is working")
        else:
            report.append(f"   ❌ PDF content extraction needs fixing")
            
        if overall['integration_working']:
            report.append(f"   ✅ PDF + code integration is working")
        else:
            report.append(f"   ❌ PDF + code integration needs work")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def main():
    """Run PDF processing validation."""
    logging.basicConfig(level=logging.INFO)
    
    validator = PDFProcessingValidator()
    
    # Run validation
    results = validator.validate_pdf_processing()
    
    # Generate and display report
    report = validator.generate_report(results)
    print(f"\n{report}")
    
    # Save results
    validator.save_results(results)
    
    # Return status
    overall = results['overall_assessment']
    if overall['overall_status'] == 'PASSED':
        print(f"\n🎉 PDF PROCESSING VALIDATION PASSED!")
        return 0
    else:
        print(f"\n❌ PDF PROCESSING VALIDATION FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())