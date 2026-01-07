import os
from typing import List, Dict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import pdfplumber
from groq import Groq
import time
import re
import shutil

# ============================================================================
# 🎯 CHONKIE INTEGRATION - Semantic Chunking for Tables
# ============================================================================
try:
    from chonkie import SemanticChunker
    CHONKIE_AVAILABLE = True
    print("✅ Chonkie available - Using semantic chunking!")
except ImportError:
    CHONKIE_AVAILABLE = False
    print("⚠️ Chonkie not found - Install: pip install chonkie")
    print("   Falling back to enhanced RecursiveCharacterTextSplitter")

# Always import RecursiveCharacterTextSplitter for fallback
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FinancialRAG:
    def __init__(self):
        print("=" * 60)
        print("📦 Initializing PRODUCTION Bilingual Financial RAG")
        print("=" * 60)
        
        # STEP 1: Load embeddings
        print("\n⚡ Loading HuggingFace embeddings...")
        start_time = time.time()
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-small",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            
            load_time = time.time() - start_time
            print(f"✅ Embeddings loaded in {load_time:.1f}s")
            
            # Test embeddings
            test_embedding = self.embeddings.embed_query("test")
            print(f"✅ Dimension: {len(test_embedding)}")
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            raise
        
        # STEP 2: Initialize Chonkie or fallback
        if CHONKIE_AVAILABLE:
            print("\n🔧 Initializing Chonkie Semantic Chunker...")
            try:
                self.chunker = SemanticChunker(
                    embedding_model="intfloat/multilingual-e5-small",
                    chunk_size=1200,
                    similarity_threshold=0.75
                )
                print("✅ Chonkie ready - Tables will stay intact!")
            except Exception as e:
                print(f"⚠️ Chonkie initialization failed: {e}")
                print("🔄 Using enhanced fallback chunking")
                self.chunker = None
        else:
            self.chunker = None
        
        # STEP 3: Initialize Groq
        print("\n⚡ Initializing Groq LLM...")
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.llm_model = "llama-3.3-70b-versatile"
        print("✅ Groq LLM ready")
        
        # STEP 4: Vector store
        self.vectorstore = None
        self.persist_directory = "chroma_db_v2"
        
        # 🔧 Clean old incompatible DB
        self._cleanup_old_chroma()
        
        print("\n" + "=" * 60)
        print("🌍 Languages: English + Arabic")
        print("📊 Table Support: ENHANCED with semantic chunking")
        print("💰 Cost: $0.00 (local embeddings + Groq)")
        print("=" * 60 + "\n")
    
    def _cleanup_old_chroma(self):
        """Remove old incompatible ChromaDB if exists"""
        if os.path.exists(self.persist_directory):
            try:
                print(f"🔍 Checking existing ChromaDB at {self.persist_directory}...")
                test_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print("✅ Existing ChromaDB is compatible")
            except Exception as e:
                error_msg = str(e)
                if "collections.topic" in error_msg or "no such column" in error_msg:
                    print(f"⚠️ Incompatible ChromaDB detected")
                    print(f"🗑️ Removing old database: {self.persist_directory}")
                    shutil.rmtree(self.persist_directory)
                    print("✅ Old database cleared")
                else:
                    print(f"⚠️ ChromaDB check warning: {e}")
    
    # ========================================================================
    # 🔧 FIX #1: Better Table Extraction with Clear Markers
    # ========================================================================
    def extract_text_with_tables(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text AND tables with ENHANCED formatting
        
        KEY CHANGES:
        - Add clear BEGIN/END markers for tables
        - Better Markdown formatting
        - Add table type detection
        """
        full_text = ""
        metadata = {
            'pages_with_tables': [],
            'total_tables': 0,
            'table_locations': [],
            'table_types': []
        }
        
        print(f"\n📊 Starting table extraction from: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Page header
                    page_content = f"\n{'='*70}\n"
                    page_content += f"PAGE {page_num}\n"
                    page_content += f"{'='*70}\n\n"
                    
                    # Extract regular text
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        page_content += page_text.strip() + "\n\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    
                    if tables and len(tables) > 0:
                        metadata['pages_with_tables'].append(page_num)
                        
                        print(f"   📊 Page {page_num}: Found {len(tables)} table(s)")
                        
                        for table_num, table in enumerate(tables, 1):
                            if not table or len(table) < 2:
                                continue
                            
                            table_id = f"Table_{page_num}_{table_num}"
                            
                            # Detect table type
                            table_type = self._detect_table_type(table)
                            
                            metadata['table_locations'].append({
                                'id': table_id,
                                'page': page_num,
                                'table_num': table_num,
                                'type': table_type
                            })
                            metadata['total_tables'] += 1
                            metadata['table_types'].append(table_type)
                            
                            # 🔥 KEY FIX: Add clear BEGIN/END markers
                            page_content += f"\n{'─'*70}\n"
                            page_content += f"📊 BEGIN {table_id} - {table_type.upper()}\n"
                            page_content += f"{'─'*70}\n\n"
                            
                            # Format as Markdown
                            page_content += self._format_table_as_markdown(table)
                            
                            page_content += f"\n{'─'*70}\n"
                            page_content += f"📊 END {table_id}\n"
                            page_content += f"{'─'*70}\n\n"
                            
                            print(f"      ✓ {table_id} ({table_type}): {len(table)} rows")
                    
                    full_text += page_content
            
            print(f"\n✅ Extraction complete:")
            print(f"   📄 Total characters: {len(full_text):,}")
            print(f"   📊 Tables found: {metadata['total_tables']}")
            print(f"   📑 Pages with tables: {len(metadata['pages_with_tables'])}")
            if metadata['table_types']:
                print(f"   🏷️  Table types: {', '.join(set(metadata['table_types']))}")
            
            return full_text, metadata
            
        except Exception as e:
            print(f"❌ pdfplumber error: {e}")
            raise
    
    def _detect_table_type(self, table: List[List]) -> str:
        """Detect what type of financial table this is"""
        if not table or len(table) < 1:
            return "unknown"
        
        # Convert first few rows to text for analysis
        table_text = " ".join([
            " ".join([str(cell) for cell in row if cell])
            for row in table[:3]
        ]).lower()
        
        # Check for table type indicators
        if any(kw in table_text for kw in ['income statement', 'net interest', 'operating profit', 'non-funded']):
            return "income_statement"
        elif any(kw in table_text for kw in ['balance sheet', 'total assets', 'gross loans', 'deposits']):
            return "balance_sheet"
        elif any(kw in table_text for kw in ['key metrics', 'npl ratio', 'cet-1', 'cost to income']):
            return "key_metrics"
        elif any(kw in table_text for kw in ['9m', 'q3', 'q2', 'yoy', 'qoq']):
            return "financial_data"
        else:
            return "general_table"
    
    def _format_table_as_markdown(self, table: List[List]) -> str:
        """
        Convert table to clean Markdown
        
        KEY CHANGES:
        - Better cell cleaning
        - Handle empty cells properly
        - Preserve numbers exactly
        """
        if not table or len(table) < 1:
            return ""
        
        markdown = ""
        cleaned_rows = []
        
        # Clean all rows
        for row in table:
            if row and any(cell for cell in row):
                clean_row = []
                for cell in row:
                    if cell is None or str(cell).strip() == "":
                        clean_row.append("—")
                    else:
                        # Preserve numbers and text exactly
                        clean_row.append(str(cell).strip())
                cleaned_rows.append(clean_row)
        
        if not cleaned_rows or len(cleaned_rows) < 2:
            return ""
        
        # Header row
        headers = cleaned_rows[0]
        markdown += "| " + " | ".join(headers) + " |\n"
        
        # Separator
        markdown += "|" + "|".join([" --- " for _ in headers]) + "|\n"
        
        # Data rows
        for row in cleaned_rows[1:]:
            # Pad row if needed
            while len(row) < len(headers):
                row.append("—")
            markdown += "| " + " | ".join(row[:len(headers)]) + " |\n"
        
        return markdown + "\n"
    
    # ========================================================================
    # 🔧 FIX #2: Better Chunking that Preserves Tables
    # ========================================================================
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Smart chunking with table preservation
        
        KEY CHANGES:
        - Use BEGIN/END markers to split
        - Add table type to metadata
        - Better fallback chunking
        """
        print("\n✂️ Starting smart chunking...")
        
        if CHONKIE_AVAILABLE and self.chunker:
            print("🔧 Using Chonkie Semantic Chunker...")
            return self._chunk_with_chonkie(text, metadata)
        else:
            print("🔧 Using enhanced table-aware chunking...")
            return self._chunk_with_fallback(text, metadata)
    
    def _chunk_with_chonkie(self, text: str, metadata: Dict) -> List[Dict]:
        """Chonkie semantic chunking"""
        try:
            chunks = self.chunker.chunk(text)
            
            chunk_dicts = []
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                
                # Detect table
                contains_table = "📊 BEGIN" in chunk_text or "Table_" in chunk_text
                
                # Extract table type if present
                table_type = "none"
                if contains_table:
                    for tt in ['income_statement', 'balance_sheet', 'key_metrics', 'financial_data']:
                        if tt in chunk_text.lower():
                            table_type = tt
                            break
                
                chunk_dicts.append({
                    'text': chunk_text,
                    'chunk_id': i,
                    'contains_table': contains_table,
                    'table_type': table_type,
                    'char_count': len(chunk_text)
                })
            
            table_chunks = sum(1 for c in chunk_dicts if c['contains_table'])
            print(f"✅ Created {len(chunk_dicts)} chunks ({table_chunks} with tables)")
            
            return chunk_dicts
            
        except Exception as e:
            print(f"⚠️ Chonkie error: {e}, using fallback...")
            return self._chunk_with_fallback(text, metadata)
    
    def _chunk_with_fallback(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Enhanced fallback chunking
        
        KEY CHANGES:
        - Split on BEGIN/END markers
        - Keep entire tables together
        - Add table type metadata
        """
        
        # 🔥 KEY FIX: Split by BEGIN/END markers
        table_pattern = r"(─{70}\n📊 BEGIN Table_\d+_\d+.*?END Table_\d+_\d+\n─{70})"
        parts = re.split(table_pattern, text, flags=re.DOTALL)
        
        chunks = []
        chunk_id = 0
        
        # Text splitter for non-table content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for part in parts:
            if not part.strip():
                continue
            
            # Check if this is a table chunk
            if "📊 BEGIN" in part:
                # Extract table type
                table_type = "unknown"
                for tt in ['income_statement', 'balance_sheet', 'key_metrics', 'financial_data']:
                    if tt.upper() in part:
                        table_type = tt
                        break
                
                chunks.append({
                    'text': part.strip(),
                    'chunk_id': chunk_id,
                    'contains_table': True,
                    'table_type': table_type,
                    'char_count': len(part)
                })
                chunk_id += 1
            else:
                # Split normal text
                sub_chunks = text_splitter.split_text(part)
                for sub_chunk in sub_chunks:
                    if sub_chunk.strip():
                        chunks.append({
                            'text': sub_chunk.strip(),
                            'chunk_id': chunk_id,
                            'contains_table': False,
                            'table_type': 'none',
                            'char_count': len(sub_chunk)
                        })
                        chunk_id += 1
        
        table_chunks = sum(1 for c in chunks if c['contains_table'])
        print(f"✅ Created {len(chunks)} chunks")
        print(f"   📊 {table_chunks} table chunks")
        print(f"   📝 {len(chunks) - table_chunks} text chunks")
        
        # 🔥 CRITICAL: Warn if no tables found
        if table_chunks == 0 and metadata.get('total_tables', 0) > 0:
            print(f"⚠️ WARNING: {metadata['total_tables']} tables extracted but 0 table chunks!")
            print("   This means chunking failed to preserve tables!")
        
        return chunks
    
    # ========================================================================
    # 🔧 FIX #3: Better Vectorstore with Proper Metadata
    # ========================================================================
    def create_vectorstore(self, chunks: List[Dict]):
        """Create vectorstore with table metadata"""
        print("\n🗄️ Creating vector database...")
    
        try:
            texts = [chunk['text'] for chunk in chunks]
        
            # Convert booleans to strings for ChromaDB
            metadatas = [
                {
                    'chunk_id': str(chunk['chunk_id']),
                    'contains_table': str(chunk['contains_table']),
                    'table_type': chunk.get('table_type', 'none'),
                    'char_count': chunk['char_count']
                }
                for chunk in chunks
            ]
        
            # Create in batches
            batch_size = 50
            total_batches = (len(texts) + batch_size - 1) // batch_size
        
            print(f"⚡ Processing {len(texts)} chunks in {total_batches} batches...")
            start_time = time.time()
        
            # First batch creates the collection
            self.vectorstore = Chroma.from_texts(
                texts=texts[:batch_size],
                embedding=self.embeddings,
                metadatas=metadatas[:batch_size],
                persist_directory=self.persist_directory,
                collection_name="financial_docs_v2"
            )
        
            # Remaining batches
            for i in range(1, total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
            
                print(f"   Batch {i+1}/{total_batches}...")
                self.vectorstore.add_texts(
                    texts[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx]
                )
        
            total_time = time.time() - start_time
            print(f"✅ Vector DB created in {total_time:.1f}s")
        
            # Verify table chunks were stored
            table_count = sum(1 for m in metadatas if m['contains_table'] == 'True')
            print(f"   📊 Stored {table_count} table chunks")
        
        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            raise
    
    # ========================================================================
    # 🔧 FIX #4: AGGRESSIVE Table Boosting in Retrieval
    # ========================================================================
    def retrieve_context(self, question: str, k: int = 5) -> List[Dict]:
        """Retrieve with AGGRESSIVE table boosting"""
        if not self.vectorstore:
            raise ValueError("No vectorstore loaded!")
    
        print(f"\n🔍 Retrieving context for: {question[:50]}...")
    
        # Detect Arabic
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)
    
        # Enhanced financial keywords
        financial_keywords = [
            'أرباح', 'ربحية', 'أصول', 'ودائع', 'فوائد', 'قروض', 'مساهمين', 
            'دخل', 'مصروفات', 'ميزانية', 'التزامات', 'income', 'revenue', 
            'profit', 'assets', 'deposits', 'interest', 'loans', 'equity', 
            'expense', 'balance', 'statement', 'total', 'grew', 'growth','npl'
        ]
    
        is_financial = any(kw in question.lower() for kw in financial_keywords)
    
        # 🔥 FIX: Retrieve MORE candidates
        search_k = 20 if is_financial else 10  # Get 20 candidates for financial queries!
    
        # Retrieve
        results = self.vectorstore.similarity_search_with_score(question, k=search_k)
    
        # 🔥 FIX: Separate tables and text
        table_results = []
        text_results = []
    
        for doc, score in results:
            item = {
                'content': doc.page_content,
                'score': score,
                'contains_table': doc.metadata.get('contains_table', False),
                'chunk_id': doc.metadata.get('chunk_id', -1)
            }
        
            if item['contains_table']:
                table_results.append(item)
            else:
                text_results.append(item)
    
        # 🔥 FIX: FORCE table inclusion for financial queries
        if is_financial and len(table_results) > 0:
            # Return MOSTLY tables for financial questions
            final_results = table_results[:k]
            # Add text only if we have room
            if len(final_results) < k:
                final_results += text_results[:(k - len(final_results))]
        else:
            # For non-financial, mix tables and text
            final_results = table_results[:2] + text_results[:3]
    
        final_results = final_results[:k]
    
        table_count = sum(1 for r in final_results if r['contains_table'])
        print(f"✅ Retrieved {len(final_results)} chunks ({table_count} with tables)")
    
        # 🚨 DEBUG: If no tables found for financial query
        if table_count == 0 and is_financial:
            print(f"⚠️ WARNING: No table chunks retrieved for financial query!")
            print(f"   Total table chunks available: {len(table_results)}")
            print(f"   Total text chunks available: {len(text_results)}")
    
        return final_results
    
    # ========================================================================
    # 🔧 FIX #5: Enhanced Prompt for Table Reading
    # ========================================================================
    def generate_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate answer with TOKEN LIMIT protection"""

        # Detect Arabic
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)

        # 🔥 FIX: Build context_text from contexts list
        # Truncate contexts to avoid token limits (Groq has ~8k context limit)
        max_context_chars = 12000  # Leave room for prompt + response
    
        context_parts = []
        total_chars = 0
    
        for i, ctx in enumerate(contexts, 1):
            content = ctx['content']
        
            # Mark if it's a table (for LLM, not shown to user)
            if ctx.get('contains_table'):
                header = f"\n[Context {i} - Contains Table Data]\n"
            else:
                header = f"\n[Context {i}]\n"
        
            chunk = header + content + "\n" + "="*50 + "\n"
        
            # Check if adding this chunk exceeds limit
            if total_chars + len(chunk) > max_context_chars:
                context_parts.append(f"\n[Additional contexts truncated...]")
                break
        
            context_parts.append(chunk)
            total_chars += len(chunk)
    
        context_text = "\n".join(context_parts)
    
        print(f"📏 Context size: {total_chars:,} characters from {len(context_parts)} contexts")

        # FIXED PROMPT - More explicit about language matching
        if is_arabic:
            prompt = f"""أنت محلل مالي خبير متخصص في التقارير المالية باللغتين العربية والإنجليزية.

    🎯 مهمتك:
    أجب على السؤال باستخدام السياقات المقدمة فقط.

    📋 تعليمات حاسمة:
    1. **أجب بالعربية مباشرة** - لا تذكر أرقام السياقات أو المصادر
    2. **الأرقام عالمية** - استخرجها من الجداول الإنجليزية
    3. **الجداول بصيغة Markdown** - الصف الأول = العناوين، الصفوف التالية = البيانات
    4. **استخرج الأرقام الدقيقة** من خلايا الجدول
    5. **احتفظ بالأرقام والمصطلحات المالية بالإنجليزية** عند الحاجة
    6. **إذا لم تجد**: قل "المعلومات غير متوفرة في الوثيقة المقدمة"
    7. **لا تقل "حسب السياق" أو "وفقاً للمصدر"** - أجب مباشرة

    💡 مثال:
    السؤال: كم بلغ إجمالي الدخل؟
    الجواب الصحيح: بلغ إجمالي الدخل 36.7 مليار درهم في الأشهر التسعة الأولى من  2025
    الجواب الخاطئ: حسب السياق 2، بلغ إجمالي الدخل...

    {context_text}

    ❓ السؤال: {question}

    💡 الجواب (مباشر وواضح):"""
    
            system_message = "أنت خبير مالي ثنائي اللغة. اقرأ الجداول الإنجليزية بعناية وأجب بالعربية. الأرقام عالمية - استخرجها من الجداول بغض النظر عن لغة السؤال."

        else:
            prompt = f"""You are an expert financial analyst specializing in Arabic and English financial reports.

    🎯 YOUR TASK:
    Answer the question using ONLY the provided contexts. Pay special attention to tables marked with 📊 [TABLE DATA].

    📋 CRITICAL INSTRUCTIONS:
    1. **Answer in ENGLISH** - the question is in English
    2. **Extract numbers from tables** - they are universal
    3. **Tables are in Markdown format** - First row = headers, rows below = data
    4. **Use exact numbers** from table cells (e.g., 36.7, 12%, 628)
    5. **Cite your sources** - mention the table or context
    6. **If not found**: Say "Information not available in the provided document"

    💡 EXAMPLE:
    Question: What was the total income?
    Answer: Total income was AED 36.7 billion in the first nine months of 2025, up 12% year-over-year according to the income statement table.

    {context_text}

    ❓ QUESTION: {question}

    💡 ANSWER (be specific, use exact numbers):"""
    
            system_message = "You are a bilingual financial expert. Read tables carefully and answer in ENGLISH. Extract exact numbers from tables."

        try:
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=800
            )
    
            return response.choices[0].message.content
    
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "413" in error_msg:
                print(f"⚠️ Token limit hit! Trying with fewer contexts...")
                if len(contexts) > 2:
                    return self.generate_answer(question, contexts[:2])
    
            print(f"❌ LLM Error: {e}")
            raise
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    def load_pdf(self, pdf_path: str) -> str:
        """Load PDF with enhanced extraction"""
        print(f"\n📄 Loading: {pdf_path}")
        text, metadata = self.extract_text_with_tables(pdf_path)
        self.doc_metadata = metadata
        return text
    
    def process_document(self, pdf_path: str) -> Dict:
        """Process document end-to-end with automatic cleanup"""
        start_time = time.time()

        try:
            # 🔥 ENHANCED: Force cleanup before processing ANY document
            print("\n🗑️ Cleaning up existing data...")
        
            # 1. Clear vector store
            if self.vectorstore is not None:
                try:
                    self.vectorstore.delete_collection()
                    print("   ✅ Vector store cleared")
                except Exception as e:
                    print(f"   ⚠️ Vector store cleanup warning: {e}")
                finally:
                    self.vectorstore = None
        
            # 2. Delete persist directory
            if os.path.exists(self.persist_directory):
                try:
                    shutil.rmtree(self.persist_directory)
                    print("   ✅ Persist directory removed")
                except Exception as e:
                    print(f"   ⚠️ Directory cleanup warning: {e}")
        
            # 3. Small delay to ensure filesystem sync
            
            time.sleep(0.5)
        
            print("✅ Cleanup complete - ready for new document\n")
        
            # Now process the new document
            text, metadata = self.extract_text_with_tables(pdf_path)
            self.doc_metadata = metadata
        
            # Chunk
            chunks = self.chunk_text(text, metadata)
        
            # Create vectorstore
            self.create_vectorstore(chunks)
        
            processing_time = time.time() - start_time
        
            print(f"\n✅ DOCUMENT PROCESSED in {processing_time:.1f}s")
            print("="*60)
        
            return {
                "status": "success",
                "chunks": len(chunks),
                "tables_found": metadata.get('total_tables', 0),
                "characters": len(text),
                "extraction_method": metadata.get('extraction_method', 'unknown'),
                "processing_time": round(processing_time, 1)
            }
        
        except Exception as e:
            print(f"❌ Processing Error: {e}")
            return {"status": "error", "error": str(e), "chunks": 0}

        
    
    def query(self, question: str) -> Dict:
        """Query the RAG system"""
        start_time = time.time()
        
        try:
            # Retrieve contexts
            contexts = self.retrieve_context(question, k=5)
            
            # Generate answer
            answer = self.generate_answer(question, contexts)
            
            query_time = time.time() - start_time
            
            # Calculate confidence
            table_contexts = sum(1 for c in contexts if c['contains_table'])
            confidence = 0.95 if table_contexts >= 2 else 0.85
            
            return {
                "answer": answer,
                "sources": [c['content'][:200] + "..." for c in contexts],
                "contexts_used": len(contexts),
                "table_contexts": table_contexts,
                "confidence": confidence,
                "query_time": round(query_time, 2)
            }
            
        except Exception as e:
            print(f"❌ Query Error: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "query_time": 0.0
            }