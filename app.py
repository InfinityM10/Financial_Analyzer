import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
from urllib.parse import urljoin
import os
import tempfile
import sys
from io import StringIO
import contextlib

# Embedding and search imports
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Financial News RAG System", 
    page_icon="📈", 
    layout="wide"
)

# Initialize session state
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None
if 'index_created' not in st.session_state:
    st.session_state.index_created = False
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'metadata' not in st.session_state:
    st.session_state.metadata = []

# Title and description
st.title("📈 Financial News RAG System")
st.markdown("A complete pipeline for scraping, indexing, and querying financial news articles")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🕷️ Web Scraping", "🗂️ Index Creation", "🔍 Query & Search", "⚙️ Settings"])

# Configuration section in settings tab
with tab4:
    st.header("Configuration Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        max_articles = st.number_input("Maximum articles to scrape", min_value=1, max_value=50, value=10)
        delay_between_requests = st.slider("Delay between requests (seconds)", 0.5, 5.0, 1.5, 0.5)
        top_k_results = st.number_input("Top K search results", min_value=1, max_value=10, value=3)
    
    with col2:
        embedding_model = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
        llama_api_url = st.text_input("LLaMA API URL", value="http://localhost:11434/api/generate")
        llama_model = st.text_input("LLaMA Model Name", value="llama3.1")

# Web Scraping Tab
with tab1:
    st.header("🕷️ Financial News Scraping")
    st.markdown("Scrape latest financial news from Economic Times")
    
    if st.button("Start Scraping", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        # Scraping configuration
        BASE_URL = "https://economictimes.indiatimes.com/"
        MARKETS_URL = "https://economictimes.indiatimes.com/markets/live-coverage"
        HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        def fetch_html(url):
            try:
                response = requests.get(url, headers=HEADERS, timeout=10)
                response.raise_for_status()
                return response.text
            except Exception as e:
                st.error(f"Failed to fetch {url}: {e}")
                return None
        
        def extract_headlines(soup):
            article_links = []
            seen_urls = set()
            for a_tag in soup.select("li.clearfix a"):
                href = a_tag.get("href")
                title = a_tag.get_text(strip=True)
                if href and "/news/" in href and "/videos/" not in href and href not in seen_urls:
                    full_url = urljoin(BASE_URL, href)
                    article_links.append((title, full_url))
                    seen_urls.add(href)
            return article_links
        
        def extract_article_content(article_url):
            html = fetch_html(article_url)
            if not html:
                return ""
            
            soup = BeautifulSoup(html, 'html.parser')
            content_div = soup.find('div', class_='article_wrapper')
            
            if content_div:
                allowed_tags = ['p', 'h1', 'h2', 'h3']
                elements = content_div.find_all(allowed_tags)
                full_text = "\n".join(
                    el.get_text(strip=True) for el in elements if el.get_text(strip=True)
                )
                return full_text
            return ""
        
        try:
            status_text.text("Fetching markets page...")
            progress_bar.progress(10)
            
            homepage_html = fetch_html(MARKETS_URL)
            if not homepage_html:
                st.error("Could not retrieve the markets page.")
            else:
                soup = BeautifulSoup(homepage_html, 'html.parser')
                headlines = extract_headlines(soup)
                
                progress_bar.progress(20)
                status_text.text(f"Found {len(headlines)} article links. Starting to scrape...")
                
                articles = []
                total_articles = min(len(headlines), max_articles)
                
                for i, (title, link) in enumerate(headlines[:max_articles]):
                    try:
                        status_text.text(f"Scraping article {i + 1}/{total_articles}: {title[:50]}...")
                        progress = 20 + (i + 1) * (70 / total_articles)
                        progress_bar.progress(int(progress))
                        
                        full_article = extract_article_content(link)
                        time.sleep(delay_between_requests)
                        
                        if not full_article.strip():
                            continue
                        
                        articles.append({
                            "title": title,
                            "link": link,
                            "content": full_article,
                            "scraped_at": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        st.warning(f"Skipped article due to error: {e}")
                        continue
                
                progress_bar.progress(90)
                status_text.text("Saving scraped data...")
                
                # Store in session state
                st.session_state.scraped_data = articles
                st.session_state.articles = articles
                
                progress_bar.progress(100)
                status_text.text(f"✅ Successfully scraped {len(articles)} articles!")
                
                # Display results
                with results_container:
                    st.success(f"Scraping completed! Found {len(articles)} articles.")
                    
                    for i, article in enumerate(articles):
                        with st.expander(f"📰 {article['title']}", expanded=False):
                            st.write(f"**Link:** {article['link']}")
                            st.write(f"**Scraped at:** {article['scraped_at']}")
                            st.write(f"**Content preview:** {article['content'][:300]}...")
                            
        except Exception as e:
            st.error(f"An error occurred during scraping: {e}")

# Index Creation Tab
with tab2:
    st.header("🗂️ FAISS Index Creation")
    st.markdown("Create searchable embeddings from scraped articles")

    if st.session_state.scraped_data is None:
        st.warning("⚠️ Please scrape articles first in the Web Scraping tab.")
    else:
        st.info(f"📊 Ready to index {len(st.session_state.scraped_data)} articles")

        if st.button("Create FAISS Index", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Loading embedding model...")
                progress_bar.progress(20)

                model = SentenceTransformer(embedding_model)

                status_text.text("Preparing article texts...")
                progress_bar.progress(30)

                articles = st.session_state.scraped_data
                
                # Filter out articles with empty content and prepare texts
                valid_articles = []
                texts = []
                
                for article in articles:
                    content = article.get("content", "").strip()
                    if content and len(content) > 10:  # Ensure content is not empty and has minimum length
                        valid_articles.append(article)
                        texts.append(content)

                if len(texts) == 0:
                    st.error("❌ No valid article content found to create embeddings. Please scrape articles with content first.")
                    st.stop()

                st.info(f"📝 Processing {len(texts)} valid articles with content...")

                status_text.text("Generating embeddings...")
                progress_bar.progress(50)

                # Generate embeddings in batches to avoid memory issues
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                    
                    # Ensure batch_embeddings is 2D
                    if batch_embeddings.ndim == 1:
                        batch_embeddings = batch_embeddings.reshape(1, -1)
                    
                    all_embeddings.append(batch_embeddings)
                    
                    # Update progress
                    progress = 50 + (i + len(batch_texts)) * 20 / len(texts)
                    progress_bar.progress(int(progress))

                # Concatenate all embeddings
                embeddings = np.vstack(all_embeddings)
                
                # Validate embeddings shape
                if embeddings.shape[0] == 0:
                    st.error("❌ No embeddings were generated. Please check your articles.")
                    st.stop()
                    
                if embeddings.shape[1] == 0:
                    st.error("❌ Invalid embedding dimension. Please check the embedding model.")
                    st.stop()

                status_text.text("Creating FAISS index...")
                progress_bar.progress(80)

                dimension = embeddings.shape[1]
                st.write(f"🔍 Creating index with {embeddings.shape[0]} embeddings of dimension {dimension}")
                
                # Create FAISS index
                index = faiss.IndexFlatL2(dimension)
                
                # Add embeddings to index
                index.add(embeddings.astype('float32'))

                # Store in session state
                st.session_state.index = index
                st.session_state.metadata = valid_articles  # Store only valid articles
                st.session_state.model = model
                st.session_state.index_created = True

                progress_bar.progress(100)
                status_text.text("✅ Index created successfully!")

                st.success(f"🎉 Successfully created FAISS index with {len(valid_articles)} articles!")
                st.info(f"📏 Embedding dimension: {dimension}")
                st.info(f"📊 Index size: {index.ntotal} vectors")

            except Exception as e:
                st.error(f"❌ Error creating index: {str(e)}")
                st.write("**Debug information:**")
                st.write(f"- Number of articles: {len(st.session_state.scraped_data) if st.session_state.scraped_data else 0}")
                
                if 'texts' in locals():
                    st.write(f"- Number of valid texts: {len(texts)}")
                if 'embeddings' in locals():
                    st.write(f"- Embeddings shape: {embeddings.shape}")
                
                # Show the full traceback for debugging
                import traceback
                st.code(traceback.format_exc())

# Query & Search Tab
with tab3:
    st.header("🔍 Query & Search Articles")
    st.markdown("Search through indexed articles and get AI-powered explanations")
    
    if not st.session_state.index_created:
        st.warning("⚠️ Please create the FAISS index first in the Index Creation tab.")
    else:
        st.success("✅ Index is ready for searching!")
        
        # Query input
        query = st.text_input("Enter your search query:", placeholder="e.g., stock market trends, inflation impact, etc.")
        
        if query and st.button("Search", type="primary"):
            try:
                # Search functionality
                model = st.session_state.model
                index = st.session_state.index
                metadata = st.session_state.metadata
                
                st.subheader("🔍 Search Results")
                
                # Embed query
                query_vector = model.encode([query], convert_to_numpy=True)
                
                # Ensure query vector is 2D and float32
                if query_vector.ndim == 1:
                    query_vector = query_vector.reshape(1, -1)
                query_vector = query_vector.astype('float32')
                
                # Search index
                D, I = index.search(query_vector, min(top_k_results, len(metadata)))
                
                # Validate search results
                if len(I) == 0 or len(I[0]) == 0:
                    st.warning("No search results found.")
                    st.stop()
                
                indices, distances = I[0], D[0]
                
                # Display results
                results = []
                for idx, dist in zip(indices, distances):
                    if 0 <= idx < len(metadata):  # Validate index bounds
                        article = metadata[idx].copy()
                        article['similarity_score'] = float(dist)
                        results.append(article)
                
                if results:
                    st.write(f"Found {len(results)} relevant articles:")
                    
                    for i, article in enumerate(results, 1):
                        with st.expander(f"🥇 Rank {i}: {article['title']}", expanded=(i==1)):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Link:** {article['link']}")
                                st.write(f"**Similarity Score:** {article['similarity_score']:.4f}")
                                st.write(f"**Content Preview:**")
                                st.write(article['content'][:400] + "...")
                            
                            with col2:
                                st.metric("Rank", f"#{i}")
                                st.metric("Score", f"{article['similarity_score']:.3f}")
                    
                    # AI Explanation Section
                    st.subheader("🧠 AI-Powered Explanation")
                    
                    if st.button("Get AI Explanation for Top Result", type="secondary"):
                        top_article = results[0]
                        
                        def generate_llama_response(prompt, max_tokens=300):
                            data = {
                                "model": llama_model,
                                "prompt": prompt,
                                "max_tokens": max_tokens,
                                "temperature": 0.4
                            }
                            headers = {"Content-Type": "application/json"}
                            
                            try:
                                response = requests.post(llama_api_url, headers=headers, json=data, stream=True)
                                if response.status_code == 200:
                                    full_response = ""
                                    for line in response.iter_lines():
                                        if line:
                                            try:
                                                line_data = json.loads(line.decode('utf-8'))
                                                if 'response' in line_data:
                                                    full_response += line_data['response']
                                                if line_data.get("done", False):
                                                    return full_response.strip()
                                            except json.JSONDecodeError:
                                                continue
                                    return full_response.strip()
                                else:
                                    return f"Error {response.status_code}: {response.text}"
                            except requests.exceptions.RequestException as e:
                                return f"Connection error: {str(e)}"
                        
                        llama_prompt = (
                            f"Explain this financial news article in detail as if summarizing for a portfolio manager:\n\n"
                            f"Title: {top_article['title']}\n\n"
                            f"Content: {top_article['content']}\n\n"
                            f"Make sure to break it down with full financial context and potential implications."
                        )
                        
                        with st.spinner("Generating AI explanation..."):
                            ai_response = generate_llama_response(llama_prompt)
                        
                        st.markdown("### 🤖 AI Analysis:")
                        st.write(ai_response)
                        
                else:
                    st.warning("No matching articles found for your query.")
                    
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Sidebar with system info
with st.sidebar:
    st.header("📊 System Status")
    
    # Status indicators
    if st.session_state.scraped_data:
        st.success(f"✅ {len(st.session_state.scraped_data)} articles scraped")
    else:
        st.error("❌ No articles scraped")
    
    if st.session_state.index_created:
        st.success("✅ FAISS index created")
        if 'index' in st.session_state:
            st.info(f"📊 Index contains {st.session_state.index.ntotal} vectors")
    else:
        st.error("❌ No index created")
    
    st.header("📝 Instructions")
    st.markdown("""
    1. **Web Scraping**: Start by scraping financial news articles
    2. **Index Creation**: Create searchable embeddings from scraped data
    3. **Query & Search**: Search through articles and get AI explanations
    4. **Settings**: Configure parameters in the Settings tab
    """)
    
    st.header("🔧 Quick Actions")
    if st.button("Clear All Data"):
        for key in ['scraped_data', 'index_created', 'articles', 'metadata', 'index', 'model']:
            if key in st.session_state:
                del st.session_state[key]
        
        # Reset to initial state
        st.session_state.scraped_data = None
        st.session_state.index_created = False
        st.session_state.articles = []
        st.session_state.metadata = []
        
        st.success("All data cleared!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("🚀 **Financial News RAG System** - Built with Streamlit, FAISS, and Sentence Transformers")