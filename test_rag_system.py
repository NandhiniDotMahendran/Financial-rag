"""
🧪 RAG SYSTEM VALIDATION SUITE
Tests your 5 critical financial questions + extraction quality
"""

import os
from rag_engine import FinancialRAG
import time
from typing import Dict, List
import re


class RAGValidator:
    def __init__(self, rag: FinancialRAG):
        self.rag = rag
        self.test_results = []
    
    # ========================================================================
    # TEST CASES - Your 5 Critical Questions
    # ========================================================================
    TEST_CASES = [
        {
            "question": "ما هي ربحية السهم الأساسية لعام 2024؟",
            "expected_answer": "3.56 درهم",
            "expected_value": 3.56,
            "tolerance": 0.01,
            "category": "Earnings Per Share"
        },
        {
            "question": "كم بلغ إجمالي ودائع العملاء في 31 ديسمبر 2024؟",
            "expected_answer": "666,777 مليون درهم",
            "expected_value": 666777,
            "tolerance": 100,
            "category": "Customer Deposits"
        },
        {
            "question": "ما هو صافي دخل الفوائد للسنة المنتهية في 31 ديسمبر 2024؟",
            "expected_answer": "26,369 مليون درهم",
            "expected_value": 26369,
            "tolerance": 10,
            "category": "Net Interest Income"
        },
        {
            "question": "كم بلغت القروض والذمم المدينة (الصافية) كما في 31 ديسمبر 2024؟",
            "expected_answer": "501,627 مليون درهم",
            "expected_value": 501627,
            "tolerance": 100,
            "category": "Loans and Receivables"
        },
        {
            "question": "ما هو إجمالي حقوق المساهمين في 31 ديسمبر 2024؟",
            "expected_answer": "126,214 مليون درهم",
            "expected_value": 126214,
            "tolerance": 100,
            "category": "Total Equity"
        }
    ]
    
    # ========================================================================
    # EXTRACTION VALIDATION
    # ========================================================================
    def validate_extraction(self, pdf_path: str) -> Dict:
        """Validate that tables are properly extracted"""
        print("\n" + "="*60)
        print("🔍 PHASE 1: EXTRACTION VALIDATION")
        print("="*60)
        
        text = self.rag.load_pdf(pdf_path)
        
        # Check for table markers
        table_count = text.count("### Table_")
        markdown_tables = text.count("| --- |")
        
        validation = {
            "total_chars": len(text),
            "tables_found": table_count,
            "markdown_formatted": markdown_tables,
            "has_page_markers": "PAGE" in text,
            "status": "PASS" if table_count > 0 else "FAIL"
        }
        
        print(f"✅ Extracted: {validation['total_chars']:,} characters")
        print(f"📊 Tables found: {validation['tables_found']}")
        print(f"✅ Markdown formatted: {validation['markdown_formatted']}")
        print(f"Status: {'✅ PASS' if validation['status'] == 'PASS' else '❌ FAIL'}")
        
        # Show sample table
        if table_count > 0:
            sample_start = text.find("### Table_")
            sample_end = text.find("[END OF Table_", sample_start) + 30
            sample = text[sample_start:sample_end]
            print(f"\n📋 Sample Table:\n{sample[:500]}...")
        
        return validation
    
    # ========================================================================
    # CHUNKING VALIDATION
    # ========================================================================
    def validate_chunking(self) -> Dict:
        """Validate that tables stay intact in chunks"""
        print("\n" + "="*60)
        print("🔍 PHASE 2: CHUNKING VALIDATION")
        print("="*60)
        
        # Inspect chunks in vector store
        if not self.rag.vectorstore:
            print("❌ No vector store found!")
            return {"status": "FAIL"}
        
        # Get all documents
        all_docs = self.rag.vectorstore.get()
        
        total_chunks = len(all_docs['ids'])
        table_chunks = sum(1 for meta in all_docs['metadatas'] if meta.get('contains_table', False))
        
        # Check for split tables (bad sign)
        documents = all_docs['documents']
        split_tables = 0
        intact_tables = 0
        
        for doc in documents:
            if "### Table_" in doc:
                if "[END OF Table_" in doc:
                    intact_tables += 1
                else:
                    split_tables += 1
        
        validation = {
            "total_chunks": total_chunks,
            "table_chunks": table_chunks,
            "intact_tables": intact_tables,
            "split_tables": split_tables,
            "status": "PASS" if split_tables == 0 else "WARNING"
        }
        
        print(f"✅ Total chunks: {validation['total_chunks']}")
        print(f"📊 Chunks with tables: {validation['table_chunks']}")
        print(f"✅ Intact tables: {validation['intact_tables']}")
        print(f"⚠️ Split tables: {validation['split_tables']}")
        print(f"Status: {'✅ PASS' if validation['status'] == 'PASS' else '⚠️ WARNING'}")
        
        return validation
    
    # ========================================================================
    # RETRIEVAL VALIDATION
    # ========================================================================
    def validate_retrieval(self, test_case: Dict) -> Dict:
        """Validate that correct context is retrieved"""
        question = test_case['question']
        
        contexts = self.rag.retrieve_context(question, k=5)
        
        # Check if any context contains table data
        has_table = any(ctx['contains_table'] for ctx in contexts)
        
        # Check if numbers are in contexts
        expected_val = test_case['expected_value']
        found_expected = False
        
        for ctx in contexts:
            # Extract all numbers from context
            numbers = re.findall(r'[\d,]+\.?\d*', ctx['content'])
            numbers = [float(n.replace(',', '')) for n in numbers if n]
            
            # Check if expected value is close to any number
            for num in numbers:
                if abs(num - expected_val) <= test_case['tolerance']:
                    found_expected = True
                    break
        
        validation = {
            "contexts_retrieved": len(contexts),
            "has_table": has_table,
            "found_expected_value": found_expected,
            "status": "PASS" if (has_table and found_expected) else "FAIL"
        }
        
        return validation
    
    # ========================================================================
    # ANSWER VALIDATION
    # ========================================================================
    def validate_answer(self, test_case: Dict, answer: str) -> Dict:
        """Validate that answer contains correct value"""
        expected_val = test_case['expected_value']
        
        # Extract all numbers from answer
        numbers = re.findall(r'[\d,]+\.?\d*', answer)
        numbers = [float(n.replace(',', '')) for n in numbers if n]
        
        # Check if expected value is in answer
        found = False
        for num in numbers:
            if abs(num - expected_val) <= test_case['tolerance']:
                found = True
                break
        
        # Check answer quality
        not_found_phrases = [
            "غير متوفرة", "not available", "لا توجد", "not found",
            "not clearly mentioned", "cannot find"
        ]
        
        answer_says_not_found = any(phrase in answer.lower() for phrase in not_found_phrases)
        
        validation = {
            "found_expected_value": found,
            "answer_length": len(answer),
            "says_not_found": answer_says_not_found,
            "status": "PASS" if (found and not answer_says_not_found) else "FAIL"
        }
        
        return validation
    
    # ========================================================================
    # RUN FULL TEST SUITE
    # ========================================================================
    def run_full_test(self, pdf_path: str):
        """Run complete validation pipeline"""
        print("\n" + "="*60)
        print("🧪 STARTING FULL RAG SYSTEM VALIDATION")
        print("="*60)
        
        overall_start = time.time()
        
        # Phase 1: Extraction
        extraction_result = self.validate_extraction(pdf_path)
        
        # Process document
        print("\n📦 Processing document...")
        process_result = self.rag.process_document(pdf_path)
        
        if process_result['status'] != 'success':
            print("❌ Document processing failed!")
            return
        
        # Phase 2: Chunking
        chunking_result = self.validate_chunking()
        
        # Phase 3-5: Test each question
        print("\n" + "="*60)
        print("🔍 PHASE 3: QUESTION VALIDATION")
        print("="*60)
        
        test_results = []
        
        for i, test_case in enumerate(self.TEST_CASES, 1):
            print(f"\n{'─'*60}")
            print(f"📝 TEST {i}/5: {test_case['category']}")
            print(f"{'─'*60}")
            print(f"❓ Question: {test_case['question']}")
            print(f"🎯 Expected: {test_case['expected_answer']}")
            
            # Test retrieval
            retrieval_result = self.validate_retrieval(test_case)
            print(f"\n🔍 Retrieval: {'✅ PASS' if retrieval_result['status'] == 'PASS' else '❌ FAIL'}")
            print(f"   - Retrieved {retrieval_result['contexts_retrieved']} contexts")
            print(f"   - Has table: {retrieval_result['has_table']}")
            print(f"   - Found expected value: {retrieval_result['found_expected_value']}")
            
            # Generate answer
            start = time.time()
            result = self.rag.query(test_case['question'])
            query_time = time.time() - start
            
            answer = result['answer']
            
            # Validate answer
            answer_result = self.validate_answer(test_case, answer)
            print(f"\n💬 Answer Generation: {'✅ PASS' if answer_result['status'] == 'PASS' else '❌ FAIL'}")
            print(f"   - Found expected value: {answer_result['found_expected_value']}")
            print(f"   - Says 'not found': {answer_result['says_not_found']}")
            print(f"   - Query time: {query_time:.2f}s")
            
            print(f"\n📄 ANSWER:\n{answer}\n")
            
            # Overall result
            overall_pass = (retrieval_result['status'] == 'PASS' and 
                          answer_result['status'] == 'PASS')
            
            test_results.append({
                'test_num': i,
                'category': test_case['category'],
                'question': test_case['question'],
                'expected': test_case['expected_answer'],
                'answer': answer,
                'retrieval_pass': retrieval_result['status'] == 'PASS',
                'answer_pass': answer_result['status'] == 'PASS',
                'overall_pass': overall_pass,
                'query_time': query_time
            })
            
            print(f"{'='*60}")
            print(f"TEST {i} RESULT: {'✅ PASS' if overall_pass else '❌ FAIL'}")
            print(f"{'='*60}")
        
        # Final summary
        total_time = time.time() - overall_start
        self._print_final_summary(test_results, extraction_result, 
                                  chunking_result, total_time)
    
    def _print_final_summary(self, test_results: List[Dict], 
                           extraction: Dict, chunking: Dict, total_time: float):
        """Print beautiful final summary"""
        print("\n" + "="*60)
        print("📊 FINAL VALIDATION REPORT")
        print("="*60)
        
        # Extraction Summary
        print(f"\n1️⃣ EXTRACTION:")
        print(f"   Status: {'✅ PASS' if extraction['status'] == 'PASS' else '❌ FAIL'}")
        print(f"   Tables found: {extraction['tables_found']}")
        
        # Chunking Summary
        print(f"\n2️⃣ CHUNKING:")
        print(f"   Status: {'✅ PASS' if chunking['status'] == 'PASS' else '⚠️ WARNING'}")
        print(f"   Intact tables: {chunking['intact_tables']}")
        print(f"   Split tables: {chunking['split_tables']}")
        
        # Test Results Summary
        passed = sum(1 for r in test_results if r['overall_pass'])
        failed = len(test_results) - passed
        
        print(f"\n3️⃣ QUESTION TESTS:")
        print(f"   Total: {len(test_results)}")
        print(f"   Passed: {passed} ✅")
        print(f"   Failed: {failed} ❌")
        print(f"   Success Rate: {(passed/len(test_results)*100):.1f}%")
        
        # Individual test breakdown
        print(f"\n   Test Breakdown:")
        for r in test_results:
            status = "✅" if r['overall_pass'] else "❌"
            print(f"   {status} Test {r['test_num']}: {r['category']}")
        
        # Performance
        avg_time = sum(r['query_time'] for r in test_results) / len(test_results)
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Avg query time: {avg_time:.2f}s")
        
        # Final verdict
        all_pass = (extraction['status'] == 'PASS' and 
                   chunking['status'] == 'PASS' and 
                   passed == len(test_results))
        
        print("\n" + "="*60)
        if all_pass:
            print("🎉 SYSTEM VALIDATION: ✅ ALL TESTS PASSED!")
            print("✅ Ready for production deployment!")
        elif passed >= 4:
            print("⚠️ SYSTEM VALIDATION: MOSTLY PASSING")
            print("🔧 Minor fixes needed - almost production ready")
        else:
            print("❌ SYSTEM VALIDATION: FAILED")
            print("🔧 Major fixes required before deployment")
        print("="*60 + "\n")


# ========================================================================
# MAIN EXECUTION
# ========================================================================
if __name__ == "__main__":
    print("\n🚀 RAG SYSTEM VALIDATOR")
    print("="*60)
    
    # Initialize RAG
    rag = FinancialRAG()
    
    # Initialize validator
    validator = RAGValidator(rag)
    
    # Run tests
    PDF_PATH = "your_emirates_nbd_report.pdf"  # Update this path
    
    if os.path.exists(PDF_PATH):
        validator.run_full_test(PDF_PATH)
    else:
        print(f"❌ PDF not found: {PDF_PATH}")
        print("Please update the PDF_PATH variable with your document path")