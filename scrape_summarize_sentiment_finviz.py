# 1. Install and Import Baseline Dependencies
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
# import re
from transformers import pipeline
import pandas as pd


# Config 
finviz_url = 'https://finviz.com/quote.ashx?t='
monitored_tickers = ['TSM','AAPL','GOOG']


# Setup Summarization Model
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


# 1. search for tables
def search_for_stock_news_tables(ticker): 
    url = finviz_url + ticker
    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')

    return news_table

news_tables = {ticker:search_for_stock_news_tables(ticker) for ticker in monitored_tickers}


# 2. parse table data
def parse_data(tables):
    # tables = news_tables['AMZN']
    parsed_data = []
    headline_previ = ''
    # Iterate through all tr tags in 'news_table'
    # read the text from each tr tag into text
    for x in tables.findAll('tr'):
        # get headline 
        headline = x.a.get_text() 
        # skip repeated news 
        if headline != headline_previ:
            headline_previ = headline
            # get url 
            url = x.a.get('href')
            # get headline 
            headline = x.a.get_text()
            # get news name
            news_name = x.span.get_text().strip()
            # splite text in the td tag into a list 
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element
            if len(date_scrape) == 1:
                time = date_scrape[0]
            # else load 'date' as the 1st element and 'time' as the second    
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            if 'https://finance.yahoo.com/news/' in url:
                parsed_data.append([date, time, news_name, headline, url])
        
    return parsed_data

parsed_news = {ticker:parse_data(news_tables[ticker]) for ticker in monitored_tickers} 



# 3. search and scrape URLs
def scrape_and_process(news_data):
    articles = []
    for i in range(len(news_data)):
        try:
            url = news_data[i][-1]
            print(url)
            if 'https://finance.yahoo.com/news/' in url:
                req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
                response = urlopen(req)   
                soup = BeautifulSoup(response, 'html.parser')
                results = soup.find_all('p')
                text = [res.text for res in results]
                words = ' '.join(text).split(' ')[:350]
                article = ' '.join(words)
                article = article.replace(u'\xa0', u' ')
                article = article.replace('\\', '')
            else:    
                article = ''
            
        except:
            article = ''
        
        articles.append(article)

    return articles

articles = {ticker:scrape_and_process(parsed_news[ticker]) for ticker in monitored_tickers} 


# 4.summarise all articles
print('Summarizing articles...')
def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors="pt")
        output = model.generate(input_ids, max_length=60, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}


# 5. adding sentiment analysis
print('Calculating sentiment...')
sentiment = pipeline("sentiment-analysis")
scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}


# 6. Exporting Results
print('Exporting results...')
def create_output_df(summaries, scores, parsed_news):
    output = []
    for ticker in monitored_tickers:
        for i in range(len(summaries[ticker])):
            output_this = [
                            ticker,
                            parsed_news[ticker][i][0],
                            parsed_news[ticker][i][1],
                            scores[ticker][i]['label'], 
                            scores[ticker][i]['score'], 
                            parsed_news[ticker][i][3], 
                            summaries[ticker][i], 
                            parsed_news[ticker][i][2], 
                            parsed_news[ticker][i][4],
                          ]
            output.append(output_this)
    # Set column names
    columns = ['Ticker','Date', 'Time', 'Sentiment', 'Sentiment Score','News Headline','Summary', 'Source','URL']
    output = pd.DataFrame(output, columns=columns)

    return output

output = create_output_df(summaries, scores, parsed_news)


 # 6.1 re-calculate 'Sentiment Score'
def recalculate_sentiment_score(df):
    if df['Sentiment'] == 'NEGATIVE':
        df['Sentiment Score'] = df['Sentiment Score'] * -1
    return df

final_output = output.apply(recalculate_sentiment_score, axis=1)

final_output.to_excel('final_output.xlsx', index=False)