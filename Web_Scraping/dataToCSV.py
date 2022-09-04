#import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
#import tensorflow.keras
#import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import json
import yfinance as yf
import os
import sys
import_path= '../ML_Prediction/Sentiment'
sys.path.insert(1, import_path)

import sentimentAnalysis
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
print("should've run")
data = open(json_file_name, 'r')


articles = json.load(data)
print('loaded the data')

stock = 'msft'
stock_data = get_stock_data(stock)
print('doing stuff still')
list_of_dates = fta.list_of_dates
dates_of_the_stock = {} #list of all the dates that the stock was traded for as a dictionary
stock_data['Date'] = stock_data.index #just adds the Date index as a coloumn

#making a dictionary with all of the dates in that the stock has
for index, date in enumerate(stock_data['Date']): #Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dates_of_the_stock[date] = [stock_data['Open'][index], stock_data['High'][index], stock_data['Low'][index], stock_data['Close'][index], stock_data['Adj Close'][index], stock_data['Volume'][index]]

print(dates_of_the_stock[0])
