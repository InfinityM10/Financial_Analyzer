# query_retriever_fixed.py - Fixed version with voice support

import requests
import json
import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import speech_recognition as sr
import pyttsx3
from threading import Thread
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths (can be overridden)
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
API_URL = 'http://localhost:11434/api/generate'

# Voice configuration
VOICE_ENABLED = True
TTS_ENGINE = None
RECOGNIZER = None
MICROPHONE = None

def initialize_voice_components():
    """Initialize voice components (TTS and STT)"""
    global TTS_ENGINE, RECOGNIZER, MICROPHONE, VOICE_ENABLED
    
    try:
        # Initialize Text-to-Speech
        TTS_ENGINE = pyttsx3.init()
        voices = TTS_ENGINE.getProperty('voices')
        if voices:
            # Set a female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    TTS_ENGINE.setProperty('voice', voice.id)
                    break
        
        # Set speech rate and volume
        TTS_ENGINE.setProperty('rate', 180)  # Speed of speech
        TTS_ENGINE.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Initialize Speech Recognition
        RECOGNIZER = sr.Recognizer()
        MICROPHONE = sr.Microphone()
        
        # Adjust for ambient noise
        with MICROPHONE as source:
            RECOGNIZER.adjust_for_ambient_noise(source, duration=1)
        
        logger.info("Voice components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize voice components: {e}")
        logger.info("Voice features will be disabled")
        VOICE_ENABLED = False
        return False

def speak_text(text):
    """Convert text to speech"""
    if not VOICE_ENABLED or not TTS_ENGINE:
        logger.warning("TTS not available")
        return False
    
    try:
        # Run TTS in a separate thread to avoid blocking
        def tts_thread():
            TTS_ENGINE.say(text)
            TTS_ENGINE.runAndWait()
        
        thread = Thread(target=tts_thread)
        thread.daemon = True
        thread.start()
        return True
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False

def listen_for_speech(timeout=10, phrase_timeout=3):
    """Convert speech to text"""
    if not VOICE_ENABLED or not RECOGNIZER or not MICROPHONE:
        logger.warning("Speech recognition not available")
        return None
    
    try:
        logger.info("Listening for speech...")
        with MICROPHONE as source:
            # Listen for audio input
            audio = RECOGNIZER.listen(source, timeout=timeout, phrase_time_limit=phrase_timeout)
        
        logger.info("Processing speech...")
        # Use Google's speech recognition
        text = RECOGNIZER.recognize_google(audio)
        logger.info(f"Recognized: {text}")
        return text
        
    except sr.WaitTimeoutError:
        logger.warning("No speech detected within timeout")
        return None
    except sr.UnknownValueError:
        logger.warning("Could not understand the audio")
        return None
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        return None
    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        return None

def load_faiss_index(index_path=INDEX_PATH):
    """Load FAISS index from file with better error handling"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    
    try:
        index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise

def load_metadata(metadata_path=METADATA_PATH):
    """Load metadata from file with better error handling"""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(metadata)} articles")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        raise

def embed_query(query, model):
    """Embed a query using sentence transformer with error handling"""
    try:
        # Force CPU usage to avoid torch.get_default_device error
        import torch
        if hasattr(torch, 'get_default_device'):
            device = torch.get_default_device()
        else:
            device = 'cpu'  # Fallback for older PyTorch versions
        
        embedding = model.encode([query], convert_to_numpy=True, device=device)
        return embedding
    except Exception as e:
        logger.error(f"Error embedding query: {e}")
        # Fallback without device specification
        try:
            return model.encode([query], convert_to_numpy=True)
        except Exception as e2:
            logger.error(f"Fallback embedding also failed: {e2}")
            raise

def search_index(query_vector, index, top_k):
    """Search FAISS index for similar vectors"""
    try:
        D, I = index.search(query_vector.astype('float32'), top_k)
        return I[0], D[0]
    except Exception as e:
        logger.error(f"Error searching index: {e}")
        raise

def get_top_k_articles(query, top_k=TOP_K, index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    """
    Main function to get top-k relevant articles for a query
    Fixed version with better error handling
    """
    try:
        logger.info(f"Searching for: {query}")
        
        # Load model with explicit device handling
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Load index and metadata
        index = load_faiss_index(index_path)
        metadata = load_metadata(metadata_path)
        
        # Embed query and search
        query_vector = embed_query(query, model)
        indices, distances = search_index(query_vector, index, top_k)

        results = []
        for idx, dist in zip(indices, distances):
            if idx < len(metadata) and idx >= 0:  # Ensure valid index
                article = metadata[idx].copy()
                article['similarity_score'] = float(dist)
                results.append(article)
        
        logger.info(f"Found {len(results)} relevant articles")
        return results
        
    except Exception as e:
        logger.error(f"Error in get_top_k_articles: {str(e)}")
        return []

def generate_llama_response(prompt, max_tokens=300, temperature=0.4, model="llama3.1"):
    """
    Generate response using Ollama's Llama model
    Enhanced with better error handling
    """
    data = {
        "model": model,
        "prompt": prompt,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        },
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response generated').strip()
        elif response.status_code == 404:
            return f"Model '{model}' not found. Please ensure Ollama is running and the model is installed."
        elif response.status_code == 429:
            time.sleep(5)
            return "Rate limit exceeded. Please try again later."
        else:
            return f"Ollama API Error {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434"
    except requests.exceptions.Timeout:
        return "Request to Ollama timed out. The model might be processing a large request."
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            return True, available_models
        return False, []
    except:
        return False, []

def voice_query_assistant():
    """Voice-enabled query assistant"""
    print("🎤 Voice-Enabled Financial Assistant")
    print("=" * 50)
    
    # Initialize voice components
    if not initialize_voice_components():
        print("⚠️ Voice features disabled. Falling back to text-only mode.")
        return interactive_query()
    
    # Check system status
    ollama_connected, available_models = check_ollama_connection()
    if not ollama_connected:
        print("⚠️ Warning: Cannot connect to Ollama. LLM analysis will not be available.")
        speak_text("Warning: AI analysis is not available.")
    else:
        print(f"✅ AI connected. Available models: {', '.join(available_models)}")
        speak_text("Financial assistant ready. How can I help you today?")
    
    # Check if index exists
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        message = "Search index not found. Please build the index first."
        print(f"❌ {message}")
        speak_text(message)
        return
    
    speak_text("You can ask questions by voice or type them. Say 'voice mode' to use speech input.")
    
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("1. Type your query")
        print("2. Say 'voice' for voice input")
        print("3. Type 'quit' to exit")
        
        user_input = input("\nEnter your choice or query: ").strip().lower()
        
        if user_input in ['quit', 'exit', 'q']:
            speak_text("Goodbye!")
            break
        
        query = None
        
        if user_input in ['voice', 'v', '2']:
            speak_text("Please speak your financial query now.")
            query = listen_for_speech(timeout=15, phrase_timeout=5)
            
            if not query:
                speak_text("I didn't hear anything. Please try again.")
                continue
                
            print(f"🎤 You said: {query}")
            
        elif user_input and user_input not in ['1']:
            query = user_input
        else:
            query = input("Enter your financial query: ").strip()
        
        if not query:
            speak_text("Please provide a valid query.")
            continue
        
        # Process the query
        print(f"\n🔍 Searching for: {query}")
        speak_text("Searching for relevant financial information.")
        
        results = get_top_k_articles(query)
        
        if not results:
            message = "No matching articles found for your query."
            print(f"❌ {message}")
            speak_text(message)
            continue
        
        # Display results
        print(f"\n📰 Top {len(results)} Matching Articles:")
        speak_text(f"Found {len(results)} relevant articles.")
        
        for i, article in enumerate(results, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   🔗 Link: {article['link']}")
            print(f"   📊 Relevance Score: {article['similarity_score']:.3f}")
            print(f"   📝 Preview: {article['content'][:200]}...")
        
        # Generate and speak AI analysis
        if ollama_connected and results:
            print(f"\n🧠 Generating AI Analysis...")
            speak_text("Generating analysis.")
            
            top_article = results[0]
            llama_prompt = (
                f"As a financial analyst, provide a concise market analysis based on this news:\n\n"
                f"Title: {top_article['title']}\n\n"
                f"Content: {top_article['content'][:1000]}\n\n"
                f"User Query: {query}\n\n"
                f"Provide key insights in 2-3 sentences suitable for voice delivery."
            )
            
            analysis = generate_llama_response(llama_prompt, max_tokens=200)
            
            print("\n" + "="*50)
            print("🤖 AI FINANCIAL ANALYSIS")
            print("="*50)
            print(analysis)
            print("="*50)
            
            # Speak the analysis
            speak_text("Here's the AI analysis:")
            speak_text(analysis)
            
            # Ask if user wants to hear article titles
            speak_text("Would you like me to read the article titles?")
            response = input("Read titles? (y/n): ").strip().lower()
            
            if response in ['y', 'yes']:
                for i, article in enumerate(results[:3], 1):
                    speak_text(f"Article {i}: {article['title']}")

def interactive_query():
    """Interactive query function for text-only mode"""
    print("=== Financial News Query System ===")
    
    # Check Ollama connection
    ollama_connected, available_models = check_ollama_connection()
    if not ollama_connected:
        print("⚠️ Warning: Cannot connect to Ollama. LLM analysis will not be available.")
    else:
        print(f"✅ Ollama connected. Available models: {', '.join(available_models)}")
    
    # Check if index exists
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print(f"❌ Search index not found. Please run retriever_agent.py first.")
        return
    
    while True:
        print("\n" + "="*50)
        query = input("Enter your financial query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not query:
            print("Please enter a valid query.")
            continue
        
        # Get articles
        results = get_top_k_articles(query)
        
        if not results:
            print("❌ No matching articles found.")
            continue
        
        # Display results (same as before)
        print(f"\n📰 Top {len(results)} Matching Articles:")
        for i, article in enumerate(results, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   🔗 Link: {article['link']}")
            print(f"   📊 Relevance Score: {article['similarity_score']:.3f}")
            print(f"   📝 Preview: {article['content'][:200]}...")
        
        # Generate LLM analysis
        if ollama_connected and results:
            print(f"\n🧠 Generating AI Analysis...")
            
            top_article = results[0]
            llama_prompt = (
                f"As a financial analyst, provide a comprehensive market analysis based on this news:\n\n"
                f"Title: {top_article['title']}\n\n"
                f"Content: {top_article['content']}\n\n"
                f"User Query: {query}\n\n"
                f"Focus on: market implications, risk assessment, investment opportunities, and key takeaways."
            )
            
            analysis = generate_llama_response(llama_prompt, max_tokens=400)
            
            print("\n" + "="*50)
            print("🤖 AI FINANCIAL ANALYSIS")
            print("="*50)
            print(analysis)
            print("="*50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--voice":
        # Voice mode
        voice_query_assistant()
    elif len(sys.argv) > 1:
        # Command line usage
        query = " ".join(sys.argv[1:])
        results = get_top_k_articles(query)
        
        if results:
            print(f"Top {len(results)} results for: {query}")
            for i, article in enumerate(results, 1):
                print(f"\n{i}. {article['title']}")
                print(f"   Score: {article['similarity_score']:.3f}")
                print(f"   Link: {article['link']}")
        else:
            print("No results found.")
    else:
        # Interactive mode choice
        print("Choose mode:")
        print("1. Text-only mode")
        print("2. Voice-enabled mode")
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == "2":
            voice_query_assistant()
        else:
            interactive_query()