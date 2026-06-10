"""
RAG Pipeline for VAR Enforcer
Uses Docling to parse FIFA Laws of the Game PDF and ChromaDB for vector storage
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
from dotenv import load_dotenv

# Docling imports
from docling.document_converter import DocumentConverter

# ChromaDB and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RAGPipeline:
    """
    RAG Pipeline for processing FIFA Laws of the Game and enabling semantic search
    """
    
    def __init__(
        self,
        pdf_url: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the RAG Pipeline
        
        Args:
            pdf_url: URL or local path to FIFA Laws PDF
            chroma_persist_dir: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the sentence-transformers model
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.pdf_url = pdf_url or os.getenv("FIFA_PDF_URL")
        self.chroma_persist_dir = chroma_persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "fifa_laws")
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", chunk_overlap))
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.document_converter = None
        
        logger.info(f"RAG Pipeline initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def _initialize_embedding_model(self):
        """Initialize the sentence-transformers embedding model"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        if self.chroma_client is None:
            logger.info(f"Initializing ChromaDB at: {self.chroma_persist_dir}")
            
            # Create persist directory if it doesn't exist
            Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "FIFA Laws of the Game for VAR decisions"}
            )
            
            logger.info(f"ChromaDB collection '{self.collection_name}' ready with {self.collection.count()} documents")
    
    def _initialize_docling(self):
        """Initialize Docling document converter"""
        if self.document_converter is None:
            logger.info("Initializing Docling document converter")
            
            # Initialize converter for Docling 2.x
            # No arguments needed - uses default configuration
            self.document_converter = DocumentConverter()
            
            logger.info("Docling converter initialized")
    
    def download_pdf(self, url: str, output_path: str) -> str:
        """
        Download PDF from URL or copy from local path
        
        Args:
            url: URL or local file path
            output_path: Where to save the PDF
            
        Returns:
            Path to the downloaded/copied PDF
        """
        try:
            # Check if it's a local file path
            if url.startswith("file://") or os.path.exists(url):
                local_path = url.replace("file://", "")
                if os.path.exists(local_path):
                    logger.info(f"Using local PDF file: {local_path}")
                    return local_path
            
            # Download from URL
            logger.info(f"Downloading PDF from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save to output path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"PDF downloaded successfully to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            raise
    
    def parse_pdf_with_docling(self, pdf_path: str) -> str:
        """
        Parse PDF using Docling to extract text
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            self._initialize_docling()
            
            logger.info(f"Parsing PDF with Docling: {pdf_path}")
            
            # Convert document using Docling 2.x API
            # Pass the file path as a string
            result = self.document_converter.convert(pdf_path)
            
            # Extract text from the document
            # For Docling 2.x, access the document and export to markdown
            text_content = result.document.export_to_markdown()
            
            logger.info(f"PDF parsed successfully. Extracted {len(text_content)} characters")
            return text_content
            
        except Exception as e:
            logger.error(f"Error parsing PDF with Docling: {str(e)}")
            raise
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks with metadata
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        text_length = len(text)
        
        # Split by paragraphs first to maintain context
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": current_chunk.strip(),
                    "metadata": {
                        "chunk_id": chunk_id,
                        "char_count": len(current_chunk),
                        "source": "fifa_laws_of_the_game"
                    }
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                chunk_id += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": current_chunk.strip(),
                "metadata": {
                    "chunk_id": chunk_id,
                    "char_count": len(current_chunk),
                    "source": "fifa_laws_of_the_game"
                }
            })
        
        logger.info(f"Text chunked into {len(chunks)} chunks")
        return chunks
    
    def add_documents_to_vectorstore(self, chunks: List[Dict[str, any]]):
        """
        Add document chunks to ChromaDB vector store
        
        Args:
            chunks: List of chunk dictionaries
        """
        try:
            self._initialize_embedding_model()
            self._initialize_chroma()
            
            logger.info(f"Adding {len(chunks)} chunks to vector store")
            
            # Prepare data for ChromaDB
            ids = [chunk["id"] for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
            
            # Add to ChromaDB in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                
                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end].tolist(),
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def query_vectorstore(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, any]]:
        """
        Query the vector store for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of relevant document chunks with metadata and scores
        """
        try:
            self._initialize_embedding_model()
            self._initialize_chroma()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    # Calculate similarity score (ChromaDB returns distances, convert to similarity)
                    distance = results['distances'][0][i]
                    similarity = 1 / (1 + distance)  # Convert distance to similarity
                    
                    if similarity >= similarity_threshold:
                        formatted_results.append({
                            "text": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "similarity_score": similarity,
                            "chunk_id": results['ids'][0][i]
                        })
            
            logger.info(f"Query returned {len(formatted_results)} results above threshold {similarity_threshold}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise
    
    def initialize_from_pdf(self, force_reload: bool = False) -> bool:
        """
        Complete pipeline: download, parse, chunk, and store PDF
        
        Args:
            force_reload: If True, reload even if collection exists
            
        Returns:
            True if successful
        """
        try:
            self._initialize_chroma()
            
            # Check if collection already has documents
            if self.collection.count() > 0 and not force_reload:
                logger.info(f"Collection already contains {self.collection.count()} documents. Skipping initialization.")
                logger.info("Use force_reload=True to reinitialize.")
                return True
            
            # Clear collection if force reload
            if force_reload and self.collection.count() > 0:
                logger.info("Force reload: Clearing existing collection")
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "FIFA Laws of the Game for VAR decisions"}
                )
            
            # Download PDF
            pdf_path = self.download_pdf(self.pdf_url, "./temp_fifa_laws.pdf")
            
            # Parse PDF
            text_content = self.parse_pdf_with_docling(pdf_path)
            
            # Chunk text
            chunks = self.chunk_text(text_content)
            
            # Add to vector store
            self.add_documents_to_vectorstore(chunks)
            
            # Clean up temp file if it was downloaded
            if pdf_path == "./temp_fifa_laws.pdf" and os.path.exists(pdf_path):
                os.remove(pdf_path)
                logger.info("Cleaned up temporary PDF file")
            
            logger.info("RAG Pipeline initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store collection
        
        Returns:
            Dictionary with collection statistics
        """
        self._initialize_chroma()
        
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.chroma_persist_dir,
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


# Convenience function for quick initialization
def create_rag_pipeline() -> RAGPipeline:
    """
    Create and return a RAG Pipeline instance with default settings
    
    Returns:
        Initialized RAGPipeline instance
    """
    return RAGPipeline()


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = create_rag_pipeline()
    
    # Initialize from PDF (only runs if collection is empty)
    pipeline.initialize_from_pdf()
    
    # Get stats
    stats = pipeline.get_collection_stats()
    print(f"\nCollection Stats: {stats}")
    
    # Example query
    query = "What are the rules for handball in the penalty area?"
    results = pipeline.query_vectorstore(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} relevant chunks:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['similarity_score']:.3f}")
        print(f"   Text: {result['text'][:200]}...")
        print()

# Made with Bob
