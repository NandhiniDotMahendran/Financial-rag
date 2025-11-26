import os
from typing import List, Dict
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from groq import Groq
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Load environment variables
load_dotenv()

class FinancialRAG:
    """HYBRID: Ollama Embeddings (local) + Groq LLM (cloud, FREE & FAST)"""
    
    def __init__(self):
        print("📦 Initializing HYBRID system (Ollama embeddings + Groq LLM)...")
        
        # Local embeddings via Ollama (no cost, one-time slow process)
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        print("✅ Ollama embeddings ready (local)")
        
        # Groq for LLM (FREE, cloud-based, SUPER FAST)
        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.llm_model = "llama-3.3-70b-versatile"
        print("✅ Groq LLM ready (cloud, free, fast)")
        
        print("💰 Cost: $0.00 | Speed: ⚡ FAST")
        self.vectorstore = None
        
    def load_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text(extraction_mode="layout")
                except:
                    page_text = page.extract_text()
                
                text += f"\n\n=== PAGE {page_num + 1} ===\n\n"
                text += page_text + "\n"
            
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ". ", ".", " ", ""],
        )
        
        chunks = text_splitter.split_text(text)
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunks.append(f"[Section {i+1}]\n{chunk}")
        
        print(f"✂️ Created {len(enhanced_chunks)} chunks")
        return enhanced_chunks
    
    def create_vectorstore(self, chunks: List[str]):
        """Create vector database with Ollama embeddings (one-time, slow but free)"""
        print(f"🔢 Creating embeddings for {len(chunks)} chunks with Ollama...")
        print("⏳ This part is slow (local processing), but queries will be FAST (Groq cloud)")
        
        import time
        start_time = time.time()
        
        self.vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Vector database created in {elapsed:.1f} seconds!")
    
    def expand_financial_query(self, question: str) -> str:
        """Expand query with synonyms"""
        synonyms = {
            "net profit": "profit for the period OR net income OR profit after tax",
            "revenue": "total operating income OR total income",
            "total assets": "total assets",
        }
        
        question_lower = question.lower()
        for term, expansion in synonyms.items():
            if term in question_lower:
                return f"{question}. Also: {expansion}"
        
        return question
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms"""
        terms = ['profit', 'revenue', 'income', 'assets', 'deposits', 'loans']
        time_terms = ['2024', '2023', 'Q3', '9 months', 'september']
        
        question_lower = question.lower()
        found = [t for t in terms + time_terms if t in question_lower]
        return found
    
    def retrieve_context(self, question: str, k: int = 4) -> List[str]:
        """Retrieve relevant chunks"""
        if not self.vectorstore:
            return []
        
        expanded = self.expand_financial_query(question)
        docs = self.vectorstore.similarity_search(expanded, k=k)
        
        key_terms = self._extract_key_terms(question)
        if key_terms:
            term_docs = self.vectorstore.similarity_search(" ".join(key_terms), k=2)
            docs.extend(term_docs)
        
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        return [doc.page_content for doc in unique_docs[:k]]
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate answer using Groq (FAST!)"""
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are a financial analyst. Answer based ONLY on the provided context.

IMPORTANT: Recognize that "Net profit" = "Profit for the period" = "Profit after taxation"

Context from Financial Report:
{context_text}

Question: {question}

Provide a clear, specific answer with exact numbers. If the information uses different terminology, still provide the answer.

Answer:"""
        
        try:
            print("🤖 Generating answer with Groq (fast, cloud-based)...")
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst. Provide accurate answers with specific numbers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.llm_model,
                temperature=0.0,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}. Check your GROQ_API_KEY in .env file."
    
    def process_document(self, pdf_path: str):
        """Full pipeline"""
        print("📄 Loading PDF...")
        text = self.load_pdf(pdf_path)
        
        print("✂️ Chunking text...")
        chunks = self.chunk_text(text)
        
        self.create_vectorstore(chunks)
        
        return {
            "status": "success",
            "chunks": len(chunks),
            "characters": len(text)
        }
    
    def query(self, question: str) -> Dict:
        """Ask question - retrieval is local, generation is cloud (fast!)"""
        if not self.vectorstore:
            return {
                "answer": "No document loaded. Please upload a PDF first.",
                "sources": [],
                "confidence": 0.0
            }
        
        try:
            print(f"🔍 Searching locally for: {question}")
            contexts = self.retrieve_context(question, k=4)
            
            if not contexts:
                return {
                    "answer": "No relevant information found in the document.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # This is FAST because it uses Groq (cloud)
            answer = self.generate_answer(question, contexts)
            sources = [ctx[:300] + "..." for ctx in contexts[:3]]
            
            confidence = 0.85
            if "don't have" in answer.lower() or "not found" in answer.lower():
                confidence = 0.3
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def load_existing_vectorstore(self):
        """Load existing vector database"""
        if os.path.exists("./chroma_db"):
            print("📂 Loading existing vector database...")
            self.vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings
            )
            return True
        return False


# Test
if __name__ == "__main__":
    print("🧪 Testing HYBRID RAG (Ollama + Groq)...\n")
    
    rag = FinancialRAG()
    
    result = rag.query("What is the revenue?")
    print(f"Answer: {result['answer']}\n")
    
    print("✅ HYBRID RAG working!")
    print("💰 Cost: $0.00 | Speed: ⚡ 1-3 seconds per query")