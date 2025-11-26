from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import shutil
from rag_engine import FinancialRAG

# Initialize FastAPI app
app = FastAPI(title="Financial RAG API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine (singleton)
rag = FinancialRAG()

# Try to load existing vectorstore on startup
@app.on_event("startup")
async def startup_event():
    if rag.load_existing_vectorstore():
        print("✅ Loaded existing vector database")
    else:
        print("⚠️ No existing database found - please upload a document")

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    language: Optional[str] = "english"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float

class UploadResponse(BaseModel):
    filename: str
    status: str
    chunks: int
    characters: int

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Financial RAG API v2.0 - Now with REAL AI!",
        "status": "healthy",
        "version": "2.0.0",
        "vectorstore_loaded": rag.vectorstore is not None
    }

# Upload PDF endpoint - NOW REAL!
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📄 File saved: {file_path}")
        
        # Process with RAG
        result = rag.process_document(file_path)
        
        return UploadResponse(
            filename=file.filename,
            status=result["status"],
            chunks=result["chunks"],
            characters=result["characters"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Query endpoint - NOW REAL!
@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    
    if not rag.vectorstore:
        raise HTTPException(
            status_code=400, 
            detail="No document loaded. Please upload a PDF first."
        )
    
    try:
        # Get answer from RAG
        result = rag.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Get current document info
@app.get("/documents")
async def get_documents():
    """Get list of uploaded documents"""
    upload_dir = "uploads"
    
    if not os.path.exists(upload_dir):
        return {
            "documents": [],
            "vectorstore_active": False
        }
    
    documents = []
    for filename in os.listdir(upload_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(upload_dir, filename)
            try:
                file_size = os.path.getsize(file_path)
                documents.append({
                    "name": filename,
                    "size_mb": round(file_size / (1024*1024), 2),
                    "status": "processed" if rag.vectorstore else "pending"
                })
            except:
                pass
    
    return {
        "documents": documents,
        "vectorstore_active": rag.vectorstore is not None
    }

# Clear vectorstore endpoint (for testing)
@app.delete("/clear")
async def clear_vectorstore():
    """Clear the vector database and uploads"""
    try:
        # Remove chroma db
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        
        # Remove uploads
        if os.path.exists("./uploads"):
            shutil.rmtree("./uploads")
        
        # Reset RAG
        rag.vectorstore = None
        rag.qa_chain = None
        
        return {"message": "Vector database and uploads cleared"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)