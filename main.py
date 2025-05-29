from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import json
import os
import io
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent Finance Assistant API with Voice",
    description="A comprehensive finance assistant with scraping, retrieval, LLM, and voice capabilities",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to import agents with better error handling
try:
    from scraping_agent import main as scrape_markets
    SCRAPING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Scraping agent import failed: {e}")
    SCRAPING_AVAILABLE = False

try:
    # Import the fixed query retriever
    from query_retriever_fixed import (
        get_top_k_articles, 
        generate_llama_response,
        initialize_voice_components,
        speak_text,
        listen_for_speech
    )
    RETRIEVAL_AVAILABLE = True
    
    # Initialize voice components
    VOICE_AVAILABLE = initialize_voice_components()
    logger.info(f"Voice components available: {VOICE_AVAILABLE}")
    
except ImportError as e:
    logger.error(f"Retrieval agent import failed: {e}")
    RETRIEVAL_AVAILABLE = False
    VOICE_AVAILABLE = False

# Voice processing imports
try:
    import speech_recognition as sr
    import pyttsx3
    import io
    import wave
    VOICE_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("Voice libraries not available. Install: pip install SpeechRecognition pyttsx3 pyaudio")
    VOICE_LIBS_AVAILABLE = False

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3
    include_llm_analysis: Optional[bool] = True
    voice_response: Optional[bool] = False

class VoiceQueryRequest(BaseModel):
    max_results: Optional[int] = 3
    include_llm_analysis: Optional[bool] = True

class TTSRequest(BaseModel):
    text: str
    voice_speed: Optional[int] = 180
    voice_volume: Optional[float] = 0.9

class ArticleResponse(BaseModel):
    title: str
    link: str
    content: str
    similarity_score: float
    scraped_at: str

class AnalysisResponse(BaseModel):
    query: str
    top_articles: List[ArticleResponse]
    llm_analysis: Optional[str] = None
    processing_time: float
    voice_response_available: Optional[bool] = False

class ScrapeRequest(BaseModel):
    max_articles: Optional[int] = 10
    source: Optional[str] = "economictimes"

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    timestamp: str

class VoiceResponse(BaseModel):
    status: str
    message: str
    audio_available: bool

# Global variables for paths
DATA_PATH = "moneycontrol_markets_news.json"
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.json"

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Agent Finance Assistant API with Voice",
        "version": "1.1.0",
        "endpoints": "/docs for API documentation",
        "scraping_available": str(SCRAPING_AVAILABLE),
        "retrieval_available": str(RETRIEVAL_AVAILABLE),
        "voice_available": str(VOICE_AVAILABLE)
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify all services"""
    services = {}
    
    # Check agent availability
    services["scraping_agent"] = "ready" if SCRAPING_AVAILABLE else "import_failed"
    services["retrieval_agent"] = "ready" if RETRIEVAL_AVAILABLE else "import_failed"
    services["voice_features"] = "ready" if VOICE_AVAILABLE else "unavailable"
    
    if RETRIEVAL_AVAILABLE:
        # Check if scraped data exists
        services["scraped_data"] = "ready" if os.path.exists(DATA_PATH) else "no_data"
        
        # Check if FAISS index exists
        services["faiss_index"] = "ready" if os.path.exists(INDEX_PATH) else "no_index"
        
        # Check if metadata exists
        services["metadata"] = "ready" if os.path.exists(METADATA_PATH) else "no_metadata"
    
    # Check Ollama availability
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        services["llama_ollama"] = "ready" if response.status_code == 200 else "unavailable"
    except:
        services["llama_ollama"] = "unavailable"
    
    overall_status = "healthy" if all(status == "ready" for status in services.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        services=services,
        timestamp=datetime.now().isoformat()
    )

@app.post("/scrape", response_model=Dict[str, Any])
async def scrape_financial_news(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Endpoint to trigger financial news scraping"""
    if not SCRAPING_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Scraping agent not available due to import error. Check logs for details."
        )
    
    try:
        logger.info(f"Starting scraping task for {request.max_articles} articles")
        
        # Run scraping in background
        background_tasks.add_task(run_scraping_task, request.max_articles)
        
        return {
            "status": "scraping_started",
            "message": f"Scraping up to {request.max_articles} articles in background",
            "estimated_time": f"{request.max_articles * 2} seconds"
        }
    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

async def run_scraping_task(max_articles: int):
    """Background task for scraping"""
    try:
        import scraping_agent
        scraping_agent.MAX_ARTICLES = max_articles
        scraping_agent.main()
        logger.info("Scraping completed successfully")
    except Exception as e:
        logger.error(f"Background scraping failed: {str(e)}")

@app.post("/build-index", response_model=Dict[str, Any])
async def build_search_index():
    """Endpoint to build FAISS search index from scraped data"""
    if not RETRIEVAL_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Retrieval agent not available. Please fix sentence-transformers compatibility issue."
        )
    
    try:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=400, detail="No scraped data found. Please run /scrape first.")
        
        logger.info("Building FAISS index...")
        
        # Import and run the retriever agent
        import retriever_agent
        retriever_agent.DATA_PATH = DATA_PATH
        retriever_agent.INDEX_PATH = INDEX_PATH
        retriever_agent.main()
        
        # Get article count
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        return {
            "status": "index_built",
            "message": "FAISS index built successfully",
            "articles_indexed": len(articles),
            "index_path": INDEX_PATH,
            "metadata_path": METADATA_PATH
        }
    except Exception as e:
        logger.error(f"Index building error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Index building failed: {str(e)}")

@app.post("/query", response_model=AnalysisResponse)
async def query_financial_assistant(request: QueryRequest):
    """Main endpoint for querying the financial assistant with optional voice response"""
    if not RETRIEVAL_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Query functionality not available. Please fix sentence-transformers compatibility issue."
        )
    
    try:
        start_time = datetime.now()
        
        # Check if index exists
        if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise HTTPException(
                status_code=400, 
                detail="Search index not found. Please run /build-index first."
            )
        
        logger.info(f"Processing query: {request.query}")
        
        # Get top matching articles
        results = get_top_k_articles(request.query)
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant articles found for your query.")
        
        # Convert to response format
        articles = []
        for article in results[:request.max_results]:
            articles.append(ArticleResponse(
                title=article['title'],
                link=article['link'],
                content=article['content'][:500] + "..." if len(article['content']) > 500 else article['content'],
                similarity_score=article['similarity_score'],
                scraped_at=article.get('scraped_at', 'unknown')
            ))
        
        # Generate LLM analysis if requested
        llm_analysis = None
        if request.include_llm_analysis and results:
            top_article = results[0]
            llama_prompt = (
                f"As a financial analyst, provide a comprehensive analysis of this market news for a portfolio manager:\n\n"
                f"Title: {top_article['title']}\n\n"
                f"Content: {top_article['content']}\n\n"
                f"Query Context: {request.query}\n\n"
                f"Focus on: market implications, risk factors, opportunities, and actionable insights."
            )
            
            llm_analysis = generate_llama_response(llama_prompt, max_tokens=400)
            
            # Generate voice response if requested
            if request.voice_response and VOICE_AVAILABLE and llm_analysis:
                try:
                    speak_text(f"Analysis for your query: {request.query}")
                    speak_text(llm_analysis)
                except Exception as e:
                    logger.error(f"Voice response error: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            query=request.query,
            top_articles=articles,
            llm_analysis=llm_analysis,
            processing_time=processing_time,
            voice_response_available=VOICE_AVAILABLE
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/voice-query", response_model=AnalysisResponse)
async def voice_query_financial_assistant(request: VoiceQueryRequest):
    """Voice-enabled query endpoint - listens for speech input"""
    if not VOICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Voice functionality not available. Please install required libraries."
        )
    
    try:
        logger.info("Listening for voice query...")
        
        # Listen for speech input
        query = listen_for_speech(timeout=15, phrase_timeout=5)
        
        if not query:
            raise HTTPException(status_code=400, detail="No speech detected or could not understand audio")
        
        logger.info(f"Voice query received: {query}")
        
        # Process the query using the regular query endpoint
        query_request = QueryRequest(
            query=query,
            max_results=request.max_results,
            include_llm_analysis=request.include_llm_analysis,
            voice_response=True  # Always provide voice response for voice queries
        )
        
        return await query_financial_assistant(query_request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")

@app.post("/speak", response_model=VoiceResponse)
async def text_to_speech(request: TTSRequest):
    """Convert text to speech"""
    if not VOICE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Voice functionality not available. Please install required libraries."
        )
    
    try:
        logger.info(f"Converting text to speech: {request.text[:50]}...")
        
        # Configure TTS settings if available
        if hasattr(speak_text, '__globals__'):
            # This is a simplified approach - you might need to modify your speak_text function
            # to accept parameters for speed and volume
            pass
        
        speak_text(request.text)
        
        return VoiceResponse(
            status="success",
            message="Text converted to speech successfully",
            audio_available=True
        )
        
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@app.post("/listen", response_model=Dict[str, str])
async def speech_to_text():
    """Convert speech to text"""
    if not VOICE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Voice functionality not available. Please install required libraries."
        )
    
    try:
        logger.info("Listening for speech input...")
        
        # Listen for speech with timeout
        text = listen_for_speech(timeout=10, phrase_timeout=3)
        
        if not text:
            raise HTTPException(status_code=400, detail="No speech detected or could not understand audio")
        
        return {
            "status": "success",
            "text": text,
            "message": "Speech converted to text successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")

@app.get("/scrape-status", response_model=Dict[str, Any])
async def get_scrape_status():
    """Get the status of scraped data"""
    try:
        if not os.path.exists(DATA_PATH):
            return {
                "status": "no_data",
                "message": "No scraped data found",
                "articles_count": 0,
                "last_updated": None
            }
        
        # Get file stats
        file_stats = os.stat(DATA_PATH)
        last_modified = datetime.fromtimestamp(file_stats.st_mtime)
        
        # Count articles
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        return {
            "status": "data_available",
            "message": "Scraped data found",
            "articles_count": len(articles),
            "last_updated": last_modified.isoformat(),
            "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2)
        }
        
    except Exception as e:
        logger.error(f"Error checking scrape status: {str(e)}")
        return {
            "status": "error",
            "message": f"Error checking data: {str(e)}",
            "articles_count": 0,
            "last_updated": None
        }

@app.get("/index-status", response_model=Dict[str, Any])
async def get_index_status():
    """Get the status of the search index"""
    try:
        index_exists = os.path.exists(INDEX_PATH)
        metadata_exists = os.path.exists(METADATA_PATH)
        
        if not (index_exists and metadata_exists):
            return {
                "status": "no_index",
                "message": "Search index not built",
                "index_ready": False
            }
        
        # Get file stats
        index_stats = os.stat(INDEX_PATH)
        metadata_stats = os.stat(METADATA_PATH)
        
        # Try to get metadata info
        try:
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            articles_indexed = len(metadata)
        except:
            articles_indexed = "unknown"
        
        return {
            "status": "index_ready",
            "message": "Search index is ready",
            "index_ready": True,
            "articles_indexed": articles_indexed,
            "index_size_mb": round(index_stats.st_size / (1024 * 1024), 2),
            "last_built": datetime.fromtimestamp(index_stats.st_mtime).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking index status: {str(e)}")
        return {
            "status": "error",
            "message": f"Error checking index: {str(e)}",
            "index_ready": False
        }

@app.get("/system-info", response_model=Dict[str, Any])
async def get_system_info():
    """Get comprehensive system information"""
    try:
        import platform
        import psutil
        
        system_info = {
            "system": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "resources": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "disk_total_gb": round(psutil.disk_usage('.').total / (1024**3), 2),
                "disk_free_gb": round(psutil.disk_usage('.').free / (1024**3), 2)
            },
            "services": {
                "scraping_available": SCRAPING_AVAILABLE,
                "retrieval_available": RETRIEVAL_AVAILABLE,
                "voice_available": VOICE_AVAILABLE,
                "voice_libs_available": VOICE_LIBS_AVAILABLE
            },
            "data_status": {
                "scraped_data_exists": os.path.exists(DATA_PATH),
                "faiss_index_exists": os.path.exists(INDEX_PATH),
                "metadata_exists": os.path.exists(METADATA_PATH)
            }
        }
        
        return system_info
        
    except ImportError:
        return {
            "message": "System info unavailable - psutil not installed",
            "services": {
                "scraping_available": SCRAPING_AVAILABLE,
                "retrieval_available": RETRIEVAL_AVAILABLE,
                "voice_available": VOICE_AVAILABLE,
                "voice_libs_available": VOICE_LIBS_AVAILABLE
            }
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System info error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )