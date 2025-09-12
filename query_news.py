from newsdataapi import NewsDataApiClient
import os
from dotenv import load_dotenv
load_dotenv()

api = NewsDataApiClient(apikey=os.getenv("NEWS_DATA_IO_KEY"))

response = api.news_api(q='pizza')

print(response)