from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os
from datetime import datetime

# Default configurations
DATA_PATH = "moneycontrol_markets_news.json"
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_articles(json_path):
    """Load articles from JSON file"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        print(f"[INFO] Loaded {len(data)} articles from {json_path}")
        return data
    except FileNotFoundError:
        print(f"[ERROR] File {json_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {json_path}: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Error loading articles: {e}")
        return []

def validate_articles(articles):
    """Validate and clean article data"""
    valid_articles = []
    
    for i, article in enumerate(articles):
        # Check required fields
        if not isinstance(article, dict):
            print(f"[WARNING] Article {i} is not a dictionary, skipping")
            continue
            
        required_fields = ['title', 'content', 'link']
        missing_fields = [field for field in required_fields if not article.get(field)]
        
        if missing_fields:
            print(f"[WARNING] Article {i} missing fields: {missing_fields}")
            continue
        
        # Clean content
        content = str(article['content']).strip()
        if len(content) < 50:  # Skip very short articles
            print(f"[WARNING] Article {i} content too short, skipping")
            continue
        
        # Ensure all fields are strings
        cleaned_article = {
            'title': str(article['title']).strip(),
            'content': content,
            'link': str(article['link']).strip(),
            'scraped_at': article.get('scraped_at', datetime.now().isoformat())
        }
        
        # Add any additional fields
        for key, value in article.items():
            if key not in cleaned_article:
                cleaned_article[key] = value
        
        valid_articles.append(cleaned_article)
    
    print(f"[INFO] Validated {len(valid_articles)} out of {len(articles)} articles")
    return valid_articles

def embed_articles(articles, model):
    """Generate embeddings for articles"""
    print("[INFO] Generating embeddings...")
    
    texts = []
    for article in articles:
        # Combine title and content for better embeddings
        combined_text = f"{article['title']} {article['content']}"
        texts.append(combined_text)
    
    try:
        embeddings = model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32  # Process in batches for better memory usage
        )
        
        print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"[ERROR] Error generating embeddings: {e}")
        return None

def save_index(embeddings, articles, index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    """Save FAISS index and metadata"""
    try:
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(index, index_path)
        print(f"[INFO] FAISS index saved to {index_path}")
        
        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Metadata saved to {metadata_path}")
        
        # Save index info
        index_info = {
            "index_path": index_path,
            "metadata_path": metadata_path,
            "dimension": dimension,
            "total_vectors": index.ntotal,
            "embedding_model": EMBEDDING_MODEL,
            "created_at": datetime.now().isoformat(),
            "articles_count": len(articles)
        }
        
        with open("index_info.json", "w", encoding="utf-8") as f:
            json.dump(index_info, f, indent=2)
        
        print(f"[SUCCESS] Indexed {len(articles)} articles successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error saving index: {e}")
        return False

def load_embedding_model(model_name=EMBEDDING_MODEL):
    """Load sentence transformer model"""
    try:
        print(f"[INFO] Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        print(f"[INFO] Model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None

def check_existing_index(index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    """Check if index already exists and is valid"""
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return False, "Index files not found"
    
    try:
        # Try to load index
        index = faiss.read_index(index_path)
        
        # Try to load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Check consistency
        if index.ntotal != len(metadata):
            return False, f"Index size ({index.ntotal}) doesn't match metadata size ({len(metadata)})"
        
        return True, f"Valid index found with {index.ntotal} vectors"
        
    except Exception as e:
        return False, f"Error validating index: {e}"

def main(data_path=None, index_path=None, metadata_path=None, force_rebuild=False):
    """Main function to build search index"""
    # Use provided paths or defaults
    data_path = data_path or DATA_PATH
    index_path = index_path or INDEX_PATH
    metadata_path = metadata_path or METADATA_PATH
    
    print("=" * 60)
    print("🏗️  BUILDING SEARCH INDEX FOR FINANCE ASSISTANT")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        print("Please run the scraping agent first to collect articles.")
        return False
    
    # Check existing index
    if not force_rebuild:
        index_exists, message = check_existing_index(index_path, metadata_path)
        if index_exists:
            print(f"[INFO] {message}")
            response = input("Index already exists. Rebuild? (y/N): ")
            if response.lower() != 'y':
                print("[INFO] Using existing index.")
                return True
    
    # Load and validate articles
    print(f"[INFO] Loading articles from {data_path}...")
    articles = load_articles(data_path)
    
    if not articles:
        print("[ERROR] No articles found to index.")
        return False
    
    # Validate articles
    articles = validate_articles(articles)
    
    if not articles:
        print("[ERROR] No valid articles found after validation.")
        return False
    
    # Load embedding model
    model = load_embedding_model(EMBEDDING_MODEL)
    if model is None:
        print("[ERROR] Failed to load embedding model.")
        return False
    
    # Generate embeddings
    embeddings = embed_articles(articles, model)
    if embeddings is None:
        print("[ERROR] Failed to generate embeddings.")
        return False
    
    # Save index
    success = save_index(embeddings, articles, index_path, metadata_path)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ INDEX BUILDING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Articles indexed: {len(articles)}")
        print(f"🔍 Index file: {index_path}")
        print(f"📝 Metadata file: {metadata_path}")
        print(f"🤖 Embedding model: {EMBEDDING_MODEL}")
        print("=" * 60)
        return True
    else:
        print("\n❌ INDEX BUILDING FAILED!")
        return False

if __name__ == "__main__":
    import sys
    
    # Handle command line arguments
    force_rebuild = "--force" in sys.argv or "-f" in sys.argv
    
    # Interactive mode if no data path provided
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and force_rebuild):
        print("🔧 Interactive Index Builder")
        print("-" * 30)
        
        # Get paths from user
        data_path = input(f"Enter DATA_PATH (default: {DATA_PATH}): ").strip()
        if not data_path:
            data_path = DATA_PATH
        
        index_path = input(f"Enter INDEX_PATH (default: {INDEX_PATH}): ").strip()
        if not index_path:
            index_path = INDEX_PATH
        
        metadata_path = input(f"Enter METADATA_PATH (default: {METADATA_PATH}): ").strip()
        if not metadata_path:
            metadata_path = METADATA_PATH
        
        main(data_path, index_path, metadata_path, force_rebuild)
    else:
        # Use defaults for API integration
        main(force_rebuild=force_rebuild)