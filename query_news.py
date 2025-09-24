from newsdataapi import NewsDataApiClient
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import trafilatura
from dotenv import load_dotenv
load_dotenv()

api = NewsDataApiClient(apikey=os.getenv("NEWS_DATA_IO_KEY"))
webdriver_path = "/home/kushal/WebDrivers/chromedriver-linux64/chromedriver"
# news data needs paid plans for getting content or ai summary or even sentiment stats. I'll just scrape the content from the links instead.

# function for getting the news articles data
def get_news_articles(query):
    response = api.news_api(query)
    articles = []
    for article in response['results']:
        articles.append({
            'title': article.get('title'),
            'url': article.get('link'),
            'source': article.get('source_id'),
            'publishedAt': article.get('pubDate')
        })
    return articles

print(get_news_articles("pizza"))

# this function is for going through the links and scraping the necessary data. Gonna use selenium for this.
def scrape_site(url): # actually takes in link
    options = Options()
    options.add_argument('--head') # put --headless in prod
    service = Service(executable_path=webdriver_path)
    driver = webdriver.Chrome(service=service,options=options)
    driver.get(url)
    html = driver.page_source
    driver.quit()
    text = trafilatura.extract(html)
    return text



