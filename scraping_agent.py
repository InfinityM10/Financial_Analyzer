import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
from urllib.parse import urljoin

# Configuration
BASE_URL = "https://economictimes.indiatimes.com/"
MARKETS_URL = "https://economictimes.indiatimes.com/markets/live-coverage"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
OUTPUT_FILE = "moneycontrol_markets_news.json"
MAX_ARTICLES = 10
DELAY_BETWEEN_REQUESTS = 1.5  # seconds

def fetch_html(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None

def extract_headlines(soup):
    article_links = []
    seen_urls = set()
    for a_tag in soup.select("li.clearfix a"):
        href = a_tag.get("href")
        title = a_tag.get_text(strip=True)
        # Skip video links
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

def main():
    homepage_html = fetch_html(MARKETS_URL)
    if not homepage_html:
        print("[ERROR] Could not retrieve the markets page.")
        return

    soup = BeautifulSoup(homepage_html, 'html.parser')
    headlines = extract_headlines(soup)
    articles = []

    for i, (title, link) in enumerate(headlines[:MAX_ARTICLES]):
        try:
            print(f"[INFO] Scraping article {i + 1}: {title}")
            full_article = extract_article_content(link)
            time.sleep(DELAY_BETWEEN_REQUESTS)

            if not full_article.strip():
                print(f"[SKIP] No content found at: {link}")
                continue

            articles.append({
                "title": title,
                "link": link,
                "content": full_article,
                "scraped_at": datetime.now().isoformat()
            })

        except Exception as e:
            print(f"[WARNING] Skipped an article due to error: {e}")
            continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Saved {len(articles)} articles to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
