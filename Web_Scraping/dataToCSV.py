import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow.keras
import pandas as pd
import numpy as np
import json
import yfinance as yf
import os
import sys
import_path= '../ML_Prediction/Sentiment'
sys.path.insert(1, import_path)

import sentimentAnalysis
import Statistics #not to be confused with the statistics module that's preinstalled, this is from Statistics.py
#making the data from the other files into a form more usuable for AI.

days_in_a_row = 7 #we are using the data from a week to try and predict if the market will go up or down
average_of_n_days = 4 #take the average price of the next n days (n = 3 here) which will be used to predict if the market will go up or down

def make_ticker(ticker: str): #this might be useless
    return yf.Ticker(ticker)

#the dates is represented as yy-mm-dd
def get_stock_data(tickers: str, period = '2y', interval = '1d'):
    return yf.download(tickers, period = period, interval = interval)
'''
def get_stock_data(tickers: str, start: str, end: str): #start and end should be yy-mm-dd
    return yf.download(tickers, start, end)
'''
json_file_name = 'days_to_articles.json'
print('going to run the formating')
import Formating_the_APIs as fta
print("should've run the formating into json")
data = open(json_file_name, 'r')


articles = json.load(data)
print('loaded the data')

stock = 'msft'
stock_data = get_stock_data(stock)
list_of_dates = fta.list_of_dates
dates_of_the_stock = {} #list of all the dates that the stock was traded for as a dictionary
stock_data['Date'] = stock_data.index #just adds the Date index as a coloumn

#making a dictionary with all of the dates in that the stock has
for index, date in enumerate(stock_data['Date']): #The date and the corresponding stock's prices for that day
    dates_of_the_stock[str(date).split(' ')[0]] = [stock_data['Open'][index], stock_data['High'][index], stock_data['Low'][index], stock_data['Close'][index], stock_data['Adj Close'][index], stock_data['Volume'][index]]

#print(list(dates_of_the_stock.keys())[0]) # earliest date

list_of_stock_dates = list(dates_of_the_stock.keys())# each dates is in the format yy-mm-dd 00:00:00
first_stock_date = list_of_stock_dates[0] #the first date in list_of_stock_dates which is the earliest date that exists
print(f'the first date recognized is {first_stock_date}')
first_stock_date = str(first_stock_date).split(' ')[0] #removes the 00:00:00 from the string
#print(first_date)
#[articles,stock prices]
articles_and_stock_price = [articles[first_stock_date], dates_of_the_stock[first_stock_date]] #this is a 2d array
starting_date_index = 0
for index, date in enumerate(list_of_dates):
    if date == first_stock_date:
        starting_date_index = index
        break

print(starting_date_index)

for index, date in enumerate(list_of_dates[starting_date_index+1:]):
    if date in list_of_stock_dates:
        if date not in articles.keys():
            print(f'{date} is not in articles')
        elif date not in dates_of_the_stock.keys():
            print(f'{date} in not in dates_of_the_stock')
        articles_and_stock_price.append([articles[date], dates_of_the_stock[date]])
    else:
        #need to get the last date
        last_values = articles_and_stock_price[-1]
        last_stock_prices = last_values[1] #the second index is the array of stock prices
        articles_and_stock_price.append([articles[date], last_stock_prices])



#now for the ai
folder_id = '1MifzRW3qeJXdPfVdb7xz8Q6ITjtG6u9A'
os.system('cd ../ML_Prediction/Sentiment/')
os.system(f'gdown --folder {folder_id}') #downloads the folder called "sentiment_model_weights" from a google drive

weights_path = '../sentiment_model_weights/cp.cpkt'
weights_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(weights_dir)

model = make_sentiment_model()
model.load_weights(latest)

def predict(sentence: str):
    return model.predict(np.array([sentence]))


sentiment_and_stock_prices = []
#I only need 'snippet', 'lead_paragraph'
for article_stock in articles_and_stock_price:
    articles = article_stock[0]
    stock_prices = article_stock[1]

    for article in articles:
        snippet = article['snippet']
        lead_paragraph = article['lead_paragraph']
        # put through the sentiment model
        sentiment_and_stock_prices.append([[predict(snippet), predict(lead_paragraph)], stock_prices])
