import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Financial RAG Assistant",
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
    </style>
""", unsafe_allow_html=True)

# API Base URL
API_URL = "http://localhost:8000"

# Header
st.markdown('<div class="main-header">💰 Financial Document RAG</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Financial Document",
        type=['pdf'],
        help="Upload quarterly reports, annual reports, or financial statements"
    )
    
    if uploaded_file:
        if st.button("🚀 Process Document"):
            with st.spinner("Processing document with AI... This may take 3-5 minutes with Ollama (it's processing locally for FREE!)"):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    # Increase timeout to 10 minutes for Ollama
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=600)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {uploaded_file.name} processed!")
                        st.info(f"📊 Created {data.get('chunks', 0)} chunks from {data.get('characters', 0):,} characters")
                        st.balloons()
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Upload failed: {error_detail}")
                except requests.exceptions.Timeout:
                    st.warning("⏰ Processing is taking longer than expected. Check backend logs - it might still be working!")
                    st.info("💡 Tip: Ollama processes locally (free!) but slower. Wait 5 mins and refresh page.")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Language selection
    st.subheader("🌐 Language")
    language = st.selectbox(
        "Select language",
        ["English", "Arabic", "Bilingual"],
        index=0
    )
    
    st.markdown("---")
    
    # Current documents
    st.subheader("📚 Loaded Documents")
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            data = response.json()
            docs = data.get("documents", [])
            vectorstore_active = data.get("vectorstore_active", False)
            
            if vectorstore_active:
                st.success("✅ RAG Active")
            else:
                st.warning("⚠️ No document loaded")
            
            for doc in docs:
                st.info(f"📄 {doc['name']}\n📊 {doc['size_mb']} MB\n🔹 {doc['status']}")
    except:
        st.error("❌ Backend not connected")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask Questions")
    
    # Initialize session state
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""
    if "clear_triggered" not in st.session_state:
        st.session_state.clear_triggered = False
    
    # Reset after clear
    if st.session_state.clear_triggered:
        st.session_state.selected_question = ""
        st.session_state.clear_triggered = False
    
    # Question input with auto-fill from example buttons
    question = st.text_area(
        "Enter your financial question:",
        value=st.session_state.selected_question if not st.session_state.clear_triggered else "",
        placeholder="e.g., What was the profit for the 9 months period?",
        height=100,
        key="question_input"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        ask_button = st.button("🔍 Ask Question", use_container_width=True)
    
    with col_btn2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
        if clear_button:
            st.session_state.selected_question = ""
            st.session_state.clear_triggered = True
            if "history" in st.session_state:
                st.session_state.history = []
            st.rerun()
    
    if ask_button and question:
        with st.spinner("🤖 AI is analyzing your document... (Ollama may take 1-2 minutes)"):
            try:
                payload = {
                    "question": question,
                    "language": language.lower()
                }
                response = requests.post(
                    f"{API_URL}/query",
                    json=payload,
                    timeout=180  # Increased from 60 to 180 seconds (3 minutes)
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Save to history
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "question": question,
                        "answer": data["answer"]
                    })
                    
                    # Display answer with styling
                    st.markdown("### 📝 Answer")
                    st.markdown(f"""
                    <div style='background-color: #f0f9ff; padding: 20px; border-radius: 10px; border-left: 4px solid #1E3A8A;'>
                        {data["answer"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence
                    col_conf1, col_conf2 = st.columns([1, 3])
                    with col_conf1:
                        st.metric("Confidence", f"{data['confidence']*100:.0f}%")
                    
                    # Display sources
                    if data["sources"]:
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(data["sources"], 1):
                            with st.expander(f"Source {i}"):
                                st.text(source)
                else:
                    st.error(f"❌ {response.json().get('detail', 'Query failed')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with col2:
    st.subheader("💡 Example Questions")
    
    examples = [
        "What was the total revenue?",
        "Show me the profit margins",
        "What are the key risks mentioned?",
        "Summarize the financial performance",
        "What is the debt-to-equity ratio?",
        "ما هو إجمالي الإيرادات؟ (Arabic)"
    ]
    
    st.markdown("Click to use:")
    
    for example in examples:
        if st.button(example, key=f"btn_{example}", use_container_width=True):
            st.session_state.selected_question = example

# Display conversation history
st.markdown("---")
st.subheader("📜 Recent Questions")

if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    for i, item in enumerate(reversed(st.session_state.history[-5:])):
        with st.expander(f"Q: {item['question'][:50]}..."):
            st.write(f"**Answer:** {item['answer']}")
else:
    st.info("No questions asked yet. Start by asking a question above!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with FastAPI + Streamlit | Dubai Portfolio Project"
    "</div>",
    unsafe_allow_html=True
)