from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import re

# https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis?library=true

test_ticker = ["NVDA", "BTC"]
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

params = {'q': 'goog'}
headers = {'User-Agent': 'Mozilla/5.0'}

def search_for_news(ticker):
    start = 0
    hrefs = []
    while start < 10:
        search_url = f"https://www.google.com/search?q=Yahoo {ticker}&tbm=nws&start={start}"
        r = requests.get(search_url, params=params, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        atags = soup.find_all('a')
        hrefs.append([link['href'] for link in atags])
        start += 10
    flat_list = [item for sublist in hrefs for item in sublist]
    return flat_list

exclude = ['maps', 'policies', 'preferences', 'accounts', 'support']
def clean_urls(urls, exclude):
    valid = []
    for url in urls:
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            valid.append(res)
    return list(set(valid))

def gather_news(urls):
    articles = []
    for url in urls:
        r = requests.get(url, params=params, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = " ".join(text).split(' ')[:250]
        article = " ".join(words)
        articles.append(article)
    return articles




def organize_sentiments(sentiments):
    tally = {"positive": 0, "neutral": 0, "negative": 0}
    for sentiment in sentiments:
        if sentiment["label"] == "positive":
            tally["positive"] += 1
        elif sentiment['label'] == "negativel":
            tally["negative"] += 1
        else:
            tally["neutral"] += 1
    return tally

# raw = search_for_news("NVDA")
# clean = clean_urls(raw, exclude)
# # articles = gather_news(clean)
# headers = gather_news_headers(clean)
# sentiments = []
# for sentence in headers:
#     res = pipe(sentence)[0]
#     print(res)
#     sentiments.append(res)
# tally = organize_sentiments(sentiments)

# print(headers)
# print(tally)

def sentiment_analysis(ticker):
    raw = search_for_news(ticker)
    clean = clean_urls(raw, exclude)
    articles = gather_news(clean)
    sentiments = []
    for sentence in articles:
        res = pipe(sentence)[0]
        sentiments.append(res)
    tally = organize_sentiments(sentiments)
    return {ticker: tally}

print(sentiment_analysis("NVDA"))
print(sentiment_analysis("BTC"))
print(sentiment_analysis("MSFT"))
print(sentiment_analysis("AAPL"))

# article = gather_news(["https://finance.yahoo.com/video/nvidia-q4-earnings-results-could-163048016.html"])
# print(article)


