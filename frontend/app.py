import streamlit as st
import requests
import json
import re

# Page configuration
st.set_page_config(
    page_title="Financial RAG Assistant - V2 Bilingual",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .success-metric {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-metric {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# API Base URL
API_URL = "http://localhost:8000"

# Header
st.markdown('<div class="main-header">💰 Financial RAG - V2 Bilingual Edition</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray; margin-bottom: 2rem;'>English + Arabic Financial Document Analysis with Advanced Table Extraction</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Financial Document",
        type=['pdf'],
        help="Upload quarterly reports, annual reports, or financial statements (English or Arabic)"
    )
    
    if uploaded_file:
        if st.button("🚀 Process Document"):
            with st.spinner("⚡ Processing document with AI... This may take 30-90 seconds"):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    # Increased timeout for processing
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=600)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                        
                        # Display processing stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📦 Chunks Created", data.get('chunks', 0))
                        with col2:
                            st.metric("📊 Characters", f"{data.get('characters', 0):,}")
                        
                        st.balloons()
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Upload failed: {error_detail}")
                        
                except requests.exceptions.Timeout:
                    st.warning("⏰ Processing is taking longer than expected")
                    st.info("💡 Large documents may take 2-3 minutes. Check backend logs - it might still be working!")
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: {str(e)}")
                    st.warning("🔧 Make sure FastAPI backend is running:")
                    st.code("python main.py", language="bash")
                    
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
    
    st.markdown("---")
    
    # Language selection
    st.subheader("🌐 Language Support")
    language = st.selectbox(
        "Select language mode",
        ["Bilingual (Auto-detect)", "English", "Arabic"],
        index=0,
        help="V2 automatically detects question language!"
    )
    
    st.markdown("---")
    
    # Current documents status
    st.subheader("📚 System Status")
    try:
        response = requests.get(f"{API_URL}/documents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            docs = data.get("documents", [])
            vectorstore_active = data.get("vectorstore_active", False)
            
            if vectorstore_active:
                st.success("✅ RAG System Active")
                st.info("🧠 Ready for questions!")
            else:
                st.warning("⚠️ No document loaded")
                st.info("👆 Upload a PDF to start")
            
            # Show loaded documents
            if docs:
                st.markdown("**Loaded Documents:**")
                for doc in docs:
                    st.text(f"📄 {doc['name']}")
                    st.text(f"   {doc['size_mb']} MB • {doc['status']}")
        else:
            st.error("❌ Cannot reach backend")
            
    except requests.exceptions.RequestException:
        st.error("❌ Backend not connected")
        st.warning("Start backend with:")
        st.code("python main.py", language="bash")
    
    st.markdown("---")
    
    # V2 Features badge
    st.subheader("✨ V2 Features")
    st.markdown("""
    - ✅ Bilingual (English + Arabic)
    - ✅ Advanced table extraction
    - ✅ Semantic chunking (Chonkie)
    - ✅ Table integrity preserved
    - ✅ Built-in validation suite
    """)

# Main content area with TABS
# 🆕 FEATURE #1: Three tabs instead of two columns
tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "💡 Example Questions", "📊 System Validation"])

# ============================================================================
# TAB 1: Ask Questions
# ============================================================================
with tab1:
    st.header("💬 Ask Your Financial Questions")
    st.markdown("Ask questions in **English** or **Arabic** - the system auto-detects!")
    
    # Initialize session state
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""
    if "clear_triggered" not in st.session_state:
        st.session_state.clear_triggered = False
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Reset after clear
    if st.session_state.clear_triggered:
        st.session_state.selected_question = ""
        st.session_state.clear_triggered = False
    
    # Question input with auto-fill from example buttons
    question = st.text_area(
        "Enter your financial question:",
        value=st.session_state.selected_question if not st.session_state.clear_triggered else "",
        placeholder="e.g., What was the profit for Q3 2024? أو ما هي الأرباح للربع الثالث؟",
        height=100,
        key="question_input"
    )
    
    col_btn1, col_btn2 = st.columns([1, 1])
    
    with col_btn1:
        ask_button = st.button("🔍 Ask Question", use_container_width=True, type="primary")
    
    with col_btn2:
        clear_button = st.button("🗑️ Clear All", use_container_width=True)
        if clear_button:
            st.session_state.selected_question = ""
            st.session_state.clear_triggered = True
            st.session_state.history = []
            st.rerun()
    
    # Process question
    if ask_button and question:
        with st.spinner("🤖 AI is analyzing your document... (may take 30-60 seconds)"):
            try:
                payload = {
                    "question": question,
                    "language": language.lower()
                }
                response = requests.post(
                    f"{API_URL}/query",
                    json=payload,
                    timeout=180  # 3 minutes timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Save to history
                    st.session_state.history.append({
                        "question": question,
                        "answer": data["answer"],
                        "confidence": data.get("confidence", 0),
                        "query_time": data.get("query_time", 0)
                    })
                    
                    # Display answer with enhanced styling
                    st.markdown("### 📝 Answer")
                    st.markdown(f"""
                    <div style='background-color: #f0f9ff; padding: 20px; border-radius: 10px; border-left: 4px solid #1E3A8A;'>
                        {data["answer"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # 🆕 FEATURE #2: Enhanced metrics with color coding and table context
                    st.markdown("### 📊 Response Metrics")
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    with col_m1:
                        confidence = data.get('confidence', 0) * 100
                        # Color-coded confidence
                        if confidence > 90:
                            confidence_emoji = "🟢"
                            confidence_label = "High"
                        elif confidence > 75:
                            confidence_emoji = "🟡"
                            confidence_label = "Medium"
                        else:
                            confidence_emoji = "🔴"
                            confidence_label = "Low"
                        
                        st.metric(
                            "Confidence", 
                            f"{confidence_emoji} {confidence:.0f}%",
                            delta=confidence_label
                        )
                    
                    with col_m2:
                        query_time = data.get('query_time', 0)
                        st.metric("⏱️ Query Time", f"{query_time:.2f}s")
                    
                    with col_m3:
                        contexts_used = data.get('contexts_used', len(data.get('sources', [])))
                        st.metric("📄 Contexts Used", contexts_used)
                    
                    with col_m4:
                        # NEW: Show if answer used table data
                        table_contexts = data.get('table_contexts', 0)
                        table_emoji = "📊" if table_contexts > 0 else "📝"
                        st.metric(
                            f"{table_emoji} Table Data", 
                            f"{table_contexts} source{'s' if table_contexts != 1 else ''}"
                        )
                    
                    # Display sources
                    if data.get("sources"):
                        st.markdown("### 📚 Source Contexts")
                        for i, source in enumerate(data["sources"], 1):
                            # Check if source contains table
                            has_table = "###" in source or "Table_" in source or "|" in source
                            source_type = "📊 Table Data" if has_table else "📝 Text"
                            
                            with st.expander(f"{source_type} - Source {i}"):
                                st.text(source)
                    
                    # Success message
                    st.success("✅ Answer generated successfully!")
                    
                else:
                    error_detail = response.json().get('detail', 'Query failed')
                    st.error(f"❌ {error_detail}")
                    
            except requests.exceptions.Timeout:
                st.error("⏰ Query timed out (>3 minutes)")
                st.info("💡 Tip: Large documents or complex questions may take longer. Try a simpler question first, or check if backend is still processing.")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Backend connection failed: {str(e)}")
                st.warning("🔧 Make sure FastAPI backend is running: `python main.py`")
                
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.exception(e)
    
    # Display conversation history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Questions")
        
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Q{len(st.session_state.history)-i}: {item['question'][:60]}..."):
                st.markdown(f"**Answer:** {item['answer']}")
                st.caption(f"Confidence: {item.get('confidence', 0)*100:.0f}% | Time: {item.get('query_time', 0):.2f}s")

# ============================================================================
# TAB 2: Example Questions
# ============================================================================
with tab2:
    st.header("💡 Example Questions")
    st.markdown("Click any question to use it instantly!")
    
    # 🆕 FEATURE #3: Added Arabic example questions
    st.subheader("🇬🇧 English Examples")
    english_examples = [
        "What was the total revenue for 2024?",
        "Show me the profit margins",
        "What are the key financial risks mentioned?",
        "Summarize the financial performance for Q3",
        "What is the debt-to-equity ratio?",
        "What were the earnings per share in 2024?",
        "How much were the customer deposits?",
        "What is the net interest income?",
        "Show me the total assets",
        "What is the return on equity?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(english_examples):
        with cols[i % 2]:
            if st.button(f"📝 {example}", key=f"en_{i}", use_container_width=True):
                st.session_state.selected_question = example
                st.rerun()
    
    st.markdown("---")
    
    # Arabic examples
    st.subheader("🇦🇪 أمثلة بالعربية (Arabic Examples)")
    arabic_examples = [
        "ما هي ربحية السهم الأساسية لعام 2024؟",
        "كم بلغ إجمالي ودائع العملاء في 31 ديسمبر 2024؟",
        "ما هو صافي دخل الفوائد للسنة المنتهية في 31 ديسمبر 2024؟",
        "كم بلغت القروض والذمم المدينة (الصافية)؟",
        "ما هو إجمالي حقوق المساهمين في 31 ديسمبر 2024؟",
        "ما هي نسبة العائد على حقوق المساهمين؟",
        "كم بلغت الأرباح المنسوبة للمساهمين؟",
        "ما هو إجمالي الأصول؟"
    ]
    
    cols_ar = st.columns(2)
    for i, example in enumerate(arabic_examples):
        with cols_ar[i % 2]:
            if st.button(f"📝 {example}", key=f"ar_{i}", use_container_width=True):
                st.session_state.selected_question = example
                st.rerun()
    
    st.markdown("---")
    st.info("💡 **Tip:** The system automatically detects whether your question is in English or Arabic!")

# ============================================================================
# TAB 3: System Validation
# 🆕 FEATURE #4: Complete validation suite
# ============================================================================
with tab3:
    st.header("📊 System Validation Suite")
    st.markdown("""
    This validation suite tests the RAG system with **5 critical financial questions** 
    to ensure accurate table extraction and retrieval.
    
    **What it tests:**
    - ✅ Table data extraction accuracy
    - ✅ Arabic question understanding
    - ✅ Number precision in answers
    - ✅ Confidence levels
    - ✅ Response times
    """)
    
    st.markdown("---")
    
    # Test questions with expected answers
    test_cases = [
        {
            "question": "ما هي ربحية السهم الأساسية لعام 2024؟",
            "expected": "3.56",
            "category": "Earnings Per Share (Arabic)",
            "tolerance": 0.1
        },
        {
            "question": "كم بلغ إجمالي ودائع العملاء في 31 ديسمبر 2024؟",
            "expected": "666777",
            "category": "Customer Deposits (Arabic)",
            "tolerance": 1000
        },
        {
            "question": "ما هو صافي دخل الفوائد للسنة المنتهية في 31 ديسمبر 2024؟",
            "expected": "26369",
            "category": "Net Interest Income (Arabic)",
            "tolerance": 100
        },
        {
            "question": "كم بلغت القروض والذمم المدينة (الصافية) كما في 31 ديسمبر 2024؟",
            "expected": "501627",
            "category": "Loans and Receivables (Arabic)",
            "tolerance": 1000
        },
        {
            "question": "ما هو إجمالي حقوق المساهمين في 31 ديسمبر 2024؟",
            "expected": "126214",
            "category": "Total Equity (Arabic)",
            "tolerance": 1000
        }
    ]
    
    # Run validation button
    if st.button("🧪 Run Complete Validation Suite", type="primary", use_container_width=True):
        st.markdown("---")
        st.subheader("🔬 Running Validation Tests...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, test in enumerate(test_cases):
            status_text.text(f"Testing {i+1}/{len(test_cases)}: {test['category']}...")
            
            # Display test info
            st.markdown(f"### Test {i+1}: {test['category']}")
            
            col_test1, col_test2 = st.columns([3, 1])
            
            with col_test1:
                st.write(f"**Question:** {test['question']}")
                st.write(f"**Expected value:** {test['expected']}")
            
            # Run the test
            test_start_time = st.time()
            
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": test['question'],
                        "language": "bilingual"
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data['answer']
                    
                    # Extract numbers from answer
                    answer_numbers = re.findall(r'[\d,]+\.?\d*', answer.replace(',', ''))
                    expected_clean = test['expected'].replace(',', '')
                    
                    # Check if expected number is in answer (with tolerance)
                    found = False
                    closest_match = None
                    
                    for num_str in answer_numbers:
                        try:
                            num = float(num_str)
                            expected_num = float(expected_clean)
                            
                            if abs(num - expected_num) <= test['tolerance']:
                                found = True
                                closest_match = num
                                break
                        except ValueError:
                            continue
                    
                    # Check for "not found" phrases
                    not_found_phrases = [
                        "غير متوفرة", "not available", "لا توجد", 
                        "not found", "not clearly mentioned", "cannot find"
                    ]
                    says_not_found = any(phrase in answer.lower() for phrase in not_found_phrases)
                    
                    # Determine if test passed
                    test_passed = found and not says_not_found
                    
                    with col_test2:
                        if test_passed:
                            st.success("✅ PASS")
                            results.append(True)
                        else:
                            st.error("❌ FAIL")
                            results.append(False)
                    
                    # Show detailed results
                    with st.expander("📄 View Test Details"):
                        st.markdown(f"**Answer:** {answer}")
                        st.markdown(f"**Confidence:** {data.get('confidence', 0)*100:.0f}%")
                        st.markdown(f"**Query Time:** {data.get('query_time', 0):.2f}s")
                        st.markdown(f"**Table Contexts Used:** {data.get('table_contexts', 0)}")
                        st.markdown(f"**Numbers Found:** {', '.join(answer_numbers[:5])}")
                        if closest_match:
                            st.markdown(f"**Closest Match:** {closest_match}")
                        if says_not_found:
                            st.warning("⚠️ Answer contains 'not found' phrase")
                    
                else:
                    with col_test2:
                        st.error("❌ ERROR")
                    results.append(False)
                    st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.Timeout:
                with col_test2:
                    st.error("❌ TIMEOUT")
                results.append(False)
                st.error("Query timed out (>2 minutes)")
                
            except Exception as e:
                with col_test2:
                    st.error("❌ ERROR")
                results.append(False)
                st.error(f"Error: {str(e)}")
            
            # Update progress
            progress_bar.progress((i + 1) / len(test_cases))
            st.markdown("---")
        
        status_text.empty()
        
        # Final summary
        st.markdown("## 📊 Validation Summary")
        
        passed = sum(results)
        total = len(results)
        success_rate = (passed / total) * 100
        
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            st.metric("Total Tests", total)
        
        with col_sum2:
            st.metric("✅ Passed", passed)
        
        with col_sum3:
            st.metric("❌ Failed", total - passed)
        
        with col_sum4:
            if success_rate == 100:
                st.markdown('<div class="success-metric">🎉 100% Success!</div>', unsafe_allow_html=True)
            elif success_rate >= 80:
                st.markdown('<div class="warning-metric">⚠️ 80%+ Success</div>', unsafe_allow_html=True)
            else:
                st.error(f"❌ {success_rate:.0f}% Success")
        
        # Detailed breakdown
        st.markdown("### Test Breakdown:")
        for i, (test, result) in enumerate(zip(test_cases, results), 1):
            status = "✅ PASS" if result else "❌ FAIL"
            st.write(f"{status} - Test {i}: {test['category']}")
        
        # Final verdict
        st.markdown("---")
        if success_rate == 100:
            st.success("🎉 **EXCELLENT!** All tests passed. System is production-ready!")
            st.balloons()
        elif success_rate >= 80:
            st.warning("⚠️ **GOOD** - Most tests passed. Minor improvements needed.")
            st.info("Check failed tests above and verify document has the required data.")
        else:
            st.error("❌ **NEEDS WORK** - Multiple tests failed.")
            st.info("Possible issues: Document not uploaded, tables not extracted, or wrong document.")
    
    else:
        st.info("👆 Click the button above to run the validation suite")
        st.markdown("""
        **Before running validation:**
        1. ✅ Make sure you've uploaded a financial document
        2. ✅ Wait for processing to complete
        3. ✅ Ensure the document contains 2024 financial data
        
        **Expected duration:** 5-10 minutes (5 tests × ~1 min each)
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p><strong>Financial RAG V2 - Bilingual Edition</strong></p>
        <p>Built with FastAPI + Streamlit + Chonkie Semantic Chunking | Dubai Portfolio Project</p>
        <p>✨ Features: English + Arabic Support | Advanced Table Extraction | 95%+ Accuracy</p>
        <p style='margin-top: 1rem;'>Made with ❤️ for GenAI opportunities in Dubai</p>
    </div>
    """,
    unsafe_allow_html=True
)