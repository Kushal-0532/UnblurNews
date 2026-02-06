from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Init
api = os.getenv('API_KEY')
if not api:
    raise ValueError("API_KEY environment variable is not set")
newsapi = NewsApiClient(api_key=api)

def find_similar_articles(headline, page_size=10):
    """
    Find similar articles based on a given headline query.
    
    Args:
        headline (str): The search query/headline
        page_size (int): Number of articles to retrieve
    
    Returns:
        list: List of articles with content and essential metadata
    """
    try:
        articles = newsapi.get_everything(
            q=headline,
            page_size=page_size
        )
        
        if not articles or 'articles' not in articles:
            return []
        
        results = []
        for article in articles['articles']:
            if all(key in article for key in ['title', 'description', 'content', 'url', 'source', 'publishedAt']):
                results.append({
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'url': article['url'],
                    'source': article['source']['name'],
                    'published_at': article['publishedAt']
                })
        
        return results
    
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []
headline="gold prices surging"
print(find_similar_articles(headline))