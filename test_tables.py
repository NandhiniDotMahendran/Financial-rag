"""
Test script to verify table extraction and Q&A
Run this AFTER uploading your PDF via the API
"""

import requests
import json

API_URL = "http://localhost:8000"

# Test questions that MUST use tables
TABLE_QUESTIONS = [
    # English - Balance Sheet
    {
        "question": "What was Emirates NBD's total gross loans as of September 2025?",
        "expected_answer_contains": ["628", "billion", "AED"],
        "must_cite_table": True
    },
    
    # English - Income Statement
    {
        "question": "What is the breakdown of income between net interest income and non-funded income in 9M 2025?",
        "expected_answer_contains": ["25.8", "10.9", "billion"],
        "must_cite_table": True
    },
    
    # English - Key Metrics
    {
        "question": "What is Emirates NBD's CET-1 ratio and NPL ratio as of September 2025?",
        "expected_answer_contains": ["14.7%", "2.5%"],
        "must_cite_table": True
    },
    
    # Arabic - Total Income
    {
        "question": "ما هو إجمالي دخل بنك الإمارات دبي الوطني خلال الأشهر التسعة الأولى من عام 2025؟",
        "expected_answer_contains": ["36.7", "مليار"],
        "must_cite_table": False  # This one can come from text too
    },
    
    # Arabic - Operating Expenses (QoQ comparison)
    {
        "question": "كم بلغت المصروفات التشغيلية في الربع الثالث من 2025 مقارنة بالربع الثاني؟",
        "expected_answer_contains": ["3.9", "3.6"],
        "must_cite_table": True
    }
]

def test_query(question: str, expected_contains: list, must_cite_table: bool):
    """Test a single question"""
    print(f"\n{'='*70}")
    print(f"❓ QUESTION: {question[:80]}...")
    print(f"{'='*70}")
    
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question}
        )
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        answer = result["answer"]
        table_contexts = result.get("table_contexts", 0)
        
        print(f"\n💡 ANSWER:")
        print(answer)
        print(f"\n📊 Table contexts used: {table_contexts}")
        print(f"🎯 Confidence: {result.get('confidence', 0)}")
        
        # Check if answer contains expected values
        passed = True
        
        for expected in expected_contains:
            if expected.lower() not in answer.lower():
                print(f"⚠️  Missing expected value: '{expected}'")
                passed = False
        
        # Check if table was cited (for table-requiring questions)
        if must_cite_table:
            cited_table = ("table" in answer.lower() or 
                          "جدول" in answer.lower() or
                          table_contexts > 0)
            
            if not cited_table:
                print(f"❌ FAILED: Question requires table but no table was cited!")
                print(f"   Table contexts: {table_contexts}")
                passed = False
            else:
                print(f"✅ Table was properly cited")
        
        if passed:
            print(f"\n✅ TEST PASSED")
        else:
            print(f"\n❌ TEST FAILED - Check the answer quality")
        
        return passed
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  TABLE EXTRACTION TEST SUITE                                  ║
║  Testing if RAG system can read financial tables correctly    ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Check if API is running
    try:
        health = requests.get(f"{API_URL}/")
        if health.status_code != 200:
            print("❌ API is not running! Start it with: python main.py")
            return
        
        # Check if document is loaded
        docs = requests.get(f"{API_URL}/documents")
        if docs.json().get("vectorstore_active") != True:
            print("❌ No document loaded! Upload a PDF first via POST /upload")
            return
        
        print("✅ API is healthy")
        print("✅ Document is loaded")
        print(f"\n🧪 Running {len(TABLE_QUESTIONS)} tests...\n")
        
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure you run: python main.py")
        return
    
    # Run tests
    results = []
    for i, test in enumerate(TABLE_QUESTIONS, 1):
        print(f"\n\n{'#'*70}")
        print(f"TEST {i}/{len(TABLE_QUESTIONS)}")
        print(f"{'#'*70}")
        
        passed = test_query(
            test["question"],
            test["expected_answer_contains"],
            test["must_cite_table"]
        )
        
        results.append({
            "test": i,
            "question": test["question"][:50],
            "passed": passed
        })
    
    # Summary
    print(f"\n\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{status} - Test {r['test']}: {r['question']}...")
    
    print(f"\n📈 Score: {passed_count}/{total_count} ({100*passed_count//total_count}%)")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Your RAG system correctly reads tables!")
    elif passed_count >= total_count * 0.6:
        print("\n⚠️  PARTIAL SUCCESS - Some tests failed, but tables are being used")
    else:
        print("\n❌ TESTS FAILED - Table extraction needs debugging")
        print("\n🔍 Debugging steps:")
        print("1. Check console output from main.py - are tables being extracted?")
        print("2. Look for '📊 Page X: Found N table(s)' messages")
        print("3. Check if table contexts > 0 in failed tests")
        print("4. Verify Markdown table formatting in chunks")

if __name__ == "__main__":
    main()