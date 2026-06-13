"""
VAR Enforcer - FastAPI Backend
Main application that integrates RAG pipeline and Granite engine
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import our modules
from app.rag_pipeline import RAGPipeline, create_rag_pipeline
from app.granite_engine import GraniteEngine, create_granite_engine, Language, VARExplanation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global instances
rag_pipeline: Optional[RAGPipeline] = None
granite_engine: Optional[GraniteEngine] = None
match_data: Optional[pd.DataFrame] = None


# Pydantic Models
class VARIncidentRequest(BaseModel):
    """Request model for VAR incident explanation"""
    incident_description: str = Field(
        ...,
        description="Description of the VAR incident",
        min_length=10,
        max_length=2000,
        example="The defender made contact with the attacker's leg while attempting to win the ball in the penalty area"
    )
    match_context: Optional[str] = Field(
        None,
        description="Optional match context (teams, tournament, etc.)",
        max_length=500,
        example="World Cup Final 2022, Argentina vs France"
    )
    language: str = Field(
        default="en",
        description="Response language (en, es, fr, pt)",
        example="en"
    )
    top_k: int = Field(
        default=5,
        description="Number of relevant rules to retrieve",
        ge=1,
        le=10
    )


class VARExplanationResponse(BaseModel):
    """Response model for VAR explanation"""
    decision_explanation: str
    rule_cited: List[str]
    controversy_score: int = Field(..., ge=1, le=10)
    confidence_score: int = Field(..., ge=1, le=100)
    consistency_note: str
    plain_language_summary: str
    language: str
    retrieved_rules_count: int
    match_context: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vector_store_initialized: bool
    vector_store_document_count: int
    granite_engine_mode: str
    granite_model_id: str
    match_data_loaded: bool
    match_data_records: int


class InitializeRequest(BaseModel):
    """Request to initialize/rebuild vector store"""
    force_reload: bool = Field(
        default=False,
        description="Force reload even if vector store exists"
    )


class InitializeResponse(BaseModel):
    """Response from initialization"""
    success: bool
    message: str
    document_count: int


class MatchConsistencyResponse(BaseModel):
    """Response for match consistency check"""
    team1: str
    team2: str
    historical_matches: List[Dict[str, Any]]
    total_matches: int
    context_summary: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting VAR Enforcer application...")
    
    global rag_pipeline, granite_engine, match_data
    
    try:
        # Initialize RAG Pipeline
        logger.info("Initializing RAG Pipeline...")
        rag_pipeline = create_rag_pipeline()
        
        # Check if we have a local PDF file
        local_pdf_path = Path("app/data/fifa_laws.pdf")
        if local_pdf_path.exists():
            logger.info(f"Found local PDF at: {local_pdf_path}")
            # Update PDF URL to use local file
            rag_pipeline.pdf_url = str(local_pdf_path.absolute())
        
        # Initialize vector store (only if empty)
        try:
            rag_pipeline.initialize_from_pdf(force_reload=False)
            stats = rag_pipeline.get_collection_stats()
            logger.info(f"Vector store ready with {stats['document_count']} documents")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            logger.warning("Vector store initialization failed, but continuing...")
        
        # Initialize Granite Engine
        logger.info("Initializing Granite Engine...")
        granite_engine = create_granite_engine()
        engine_info = granite_engine.get_engine_info()
        logger.info(f"Granite Engine mode: {'DEMO' if engine_info['demo_mode'] else 'LIVE'}")
        
        # Load match data
        logger.info("Loading match data...")
        match_data_path = Path("app/data/results.csv")
        if match_data_path.exists():
            try:
                match_data = pd.read_csv(match_data_path)
                logger.info(f"Loaded {len(match_data)} match records")
            except Exception as e:
                logger.error(f"Error loading match data: {str(e)}")
                match_data = None
        else:
            logger.warning(f"Match data file not found at: {match_data_path}")
            match_data = None
        
        logger.info("VAR Enforcer application started successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down VAR Enforcer application...")


# Create FastAPI app
app = FastAPI(
    title="VAR Enforcer API",
    description="AI-powered FIFA World Cup VAR decision explainer using IBM Granite and RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None
        ).dict()
    )


# API Endpoints

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the main HTML page"""
    html_path = Path("app/static/index.html")
    if html_path.exists():
        return FileResponse(html_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend not found"
        )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns system status and component information
    """
    try:
        # Check RAG pipeline
        vector_store_initialized = False
        document_count = 0
        if rag_pipeline:
            try:
                stats = rag_pipeline.get_collection_stats()
                vector_store_initialized = stats.get('document_count', 0) > 0
                document_count = stats.get('document_count', 0)
            except Exception as e:
                logger.error(f"Error getting vector store stats: {str(e)}")
        
        # Check Granite engine
        granite_mode = "unknown"
        granite_model = "unknown"
        if granite_engine:
            try:
                info = granite_engine.get_engine_info()
                granite_mode = "demo" if info.get('demo_mode', True) else "live"
                # Handle different key names: model_id (old) or model_name (Ollama)
                granite_model = info.get('model_id') or info.get('model_name', 'unknown')
            except Exception as e:
                logger.error(f"Error getting granite engine info: {str(e)}")
        
        # Check match data
        match_data_loaded = match_data is not None
        match_records = len(match_data) if match_data is not None else 0
        
        return HealthResponse(
            status="healthy",
            vector_store_initialized=vector_store_initialized,
            vector_store_document_count=document_count,
            granite_engine_mode=granite_mode,
            granite_model_id=granite_model,
            match_data_loaded=match_data_loaded,
            match_data_records=match_records
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/api/explain", response_model=VARExplanationResponse)
async def explain_var_decision(request: VARIncidentRequest):
    """
    Explain a VAR decision using RAG + Granite
    
    This endpoint:
    1. Retrieves relevant FIFA rules from vector store
    2. Generates explanation using Granite LLM
    3. Returns structured response with controversy score
    """
    try:
        # Validate components
        if not rag_pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not initialized"
            )
        
        if not granite_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Granite engine not initialized"
            )
        
        # Validate language
        try:
            language = Language(request.language)
            logger.info(f"Language requested: {request.language}")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid language. Supported: {[lang.value for lang in Language]}"
            )
        
        # Build full query with context
        full_query = request.incident_description
        if request.match_context:
            full_query = f"Match Context: {request.match_context}\n\nIncident: {request.incident_description}"
        
        logger.info(f"Processing VAR explanation request in {language.value}: {request.incident_description[:100]}...")
        
        # Step 1: Retrieve relevant rules from vector store
        logger.info("Retrieving relevant FIFA rules...")
        context_chunks = rag_pipeline.query_vectorstore(
            query=full_query,
            top_k=request.top_k,
            similarity_threshold=0.3
        )
        
        if not context_chunks:
            logger.warning("No relevant rules found in vector store")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant FIFA rules found for this incident"
            )
        
        logger.info(f"Retrieved {len(context_chunks)} relevant rule chunks")
        
        # Step 2: Generate explanation using Granite
        logger.info("Generating explanation with Granite...")
        explanation = granite_engine.explain_var_decision(
            query=full_query,
            context_chunks=context_chunks,
            language=language
        )
        
        # Step 3: Build response
        response = VARExplanationResponse(
            decision_explanation=explanation.decision_explanation,
            rule_cited=explanation.rule_cited,
            controversy_score=explanation.controversy_score,
            confidence_score=explanation.confidence_score,
            consistency_note=explanation.consistency_note,
            plain_language_summary=explanation.plain_language_summary,
            language=explanation.language,
            retrieved_rules_count=len(context_chunks),
            match_context=request.match_context
        )
        
        logger.info(f"Successfully generated explanation with controversy score: {explanation.controversy_score}, confidence: {explanation.confidence_score}%")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining VAR decision: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating explanation: {str(e)}"
        )


@app.get("/api/consistency/{team1}/{team2}", response_model=MatchConsistencyResponse)
async def get_match_consistency(team1: str, team2: str, limit: int = 10):
    """
    Get historical match data between two teams for consistency analysis
    
    Args:
        team1: First team name
        team2: Second team name
        limit: Maximum number of matches to return
    """
    try:
        if match_data is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Match data not loaded"
            )
        
        logger.info(f"Searching for matches between {team1} and {team2}")
        
        # Search for matches (case-insensitive)
        team1_lower = team1.lower()
        team2_lower = team2.lower()
        
        # Filter matches where both teams played
        matches = match_data[
            (
                (match_data['home_team'].str.lower() == team1_lower) & 
                (match_data['away_team'].str.lower() == team2_lower)
            ) | 
            (
                (match_data['home_team'].str.lower() == team2_lower) & 
                (match_data['away_team'].str.lower() == team1_lower)
            )
        ]
        
        # Limit results
        matches = matches.head(limit)
        
        # Convert to list of dicts
        match_list = matches.to_dict('records')
        
        # Generate context summary
        total_matches = len(match_list)
        if total_matches > 0:
            context_summary = f"Found {total_matches} historical match(es) between {team1} and {team2}. "
            context_summary += f"This historical data can provide context for VAR decision consistency."
        else:
            context_summary = f"No historical matches found between {team1} and {team2} in the database."
        
        logger.info(f"Found {total_matches} matches")
        
        return MatchConsistencyResponse(
            team1=team1,
            team2=team2,
            historical_matches=match_list,
            total_matches=total_matches,
            context_summary=context_summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving match consistency: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving match data: {str(e)}"
        )


@app.post("/api/initialize", response_model=InitializeResponse)
async def initialize_vector_store(request: InitializeRequest):
    """
    Manually trigger vector store initialization/rebuild
    
    This will:
    1. Parse the FIFA Laws PDF
    2. Chunk the text
    3. Generate embeddings
    4. Store in ChromaDB
    """
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not initialized"
            )
        
        logger.info(f"Manual vector store initialization requested (force_reload={request.force_reload})")
        
        # Initialize/rebuild vector store
        success = rag_pipeline.initialize_from_pdf(force_reload=request.force_reload)
        
        # Get stats
        stats = rag_pipeline.get_collection_stats()
        document_count = stats['document_count']
        
        message = "Vector store initialized successfully"
        if request.force_reload:
            message = "Vector store rebuilt successfully"
        elif document_count > 0:
            message = "Vector store already initialized (use force_reload=true to rebuild)"
        
        logger.info(f"Initialization complete: {document_count} documents")
        
        return InitializeResponse(
            success=success,
            message=message,
            document_count=document_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Initialization failed: {str(e)}"
        )


@app.get("/api/rules/search")
async def search_rules(query: str, top_k: int = 5):
    """
    Search FIFA rules directly without generating explanation
    
    Args:
        query: Search query
        top_k: Number of results to return
    """
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not initialized"
            )
        
        logger.info(f"Searching rules for: {query[:100]}...")
        
        # Query vector store
        results = rag_pipeline.query_vectorstore(
            query=query,
            top_k=top_k,
            similarity_threshold=0.3
        )
        
        # Format results
        formatted_results = [
            {
                "text": result['text'],
                "similarity_score": result['similarity_score'],
                "chunk_id": result['chunk_id'],
                "metadata": result['metadata']
            }
            for result in results
        ]
        
        return {
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching rules: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {}
        
        # RAG pipeline stats
        if rag_pipeline:
            stats['rag_pipeline'] = rag_pipeline.get_collection_stats()
        
        # Granite engine stats
        if granite_engine:
            stats['granite_engine'] = granite_engine.get_engine_info()
        
        # Match data stats
        if match_data is not None:
            stats['match_data'] = {
                "total_records": len(match_data),
                "columns": list(match_data.columns),
                "date_range": {
                    "earliest": str(match_data['date'].min()) if 'date' in match_data.columns else None,
                    "latest": str(match_data['date'].max()) if 'date' in match_data.columns else None
                }
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )


# Mount static files (for CSS, JS, images)
static_path = Path("app/static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

# Made with Bob
