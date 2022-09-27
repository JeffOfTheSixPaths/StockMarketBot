from base64 import standard_b64decode
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

stock = 'msft' #change the stock to track here
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
first_stock_date = str(first_stock_date).split(' ')[0] #removes the " 00:00:00" from the timestamp

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
weights_dir = os.path.dirname(weights_path)
latest = tf.train.latest_checkpoint(weights_dir)

train_text = 'sentiment_model_weights/sentiment_text.csv'
train_sentiment = 'sentiment_model_weights/sentiment_sentiment.npy'
s140_text = 'sentiment_model_weights/sentiment140_text.csv'
s140_sentiment = 'sentiment_model_weights/sentiment140_sentiment.npy'

train_text = pd.read_csv(train_text)
train_sentiment = np.load(train_sentiment)
train_text.pop('Unnamed: 0')
s140_text = pd.read_csv(s140_text)
s140_sentiment = np.load(s140_sentiment)
s140_text.pop('Unnamed: 0')

train_text = tf.convert_to_tensor(train_text)
s140_tensor = tf.convert_to_tensor(s140_text)
train_text = tf.concat([train_text,s140_tensor], 0)

#print(train_text.shape)
train_sentiment = tf.convert_to_tensor(train_sentiment)
train_dataset = tf.data.Dataset.from_tensor_slices((train_text,train_sentiment))
print('creating the model and encoding the data')
model = sentimentAnalysis.make_sentiment_model(train_dataset)
print('loading weights')
model.load_weights(latest)
print('loaded weights')
def predict(sentence: str):
    return model.predict(np.array([sentence]))

test_sentences = [
                    'I love this',
                    'I hate this',
                    'Gamers are oppressed',
                    'I fucking love dogs', #using the f-word here as it can be used with different connotations in different sentences
                    'I fucking hate dogs', #it can also be used as a stronger form of "really"
                    'I really love dogs',
                    'I really hate dogs',
                    'I love dogs',
                    'I hate dogs',
                    'Rubiks cubes are a neat little thing',
                    'I really like eating meat'
                ]

print('Printing test sentences \n')
for sentence in test_sentences: #just to test if the model loaded correctly, it should predict on the interval [0,1], but it frequently leaves that interval
    print(sentence)
    print(predict(sentence))
    print('\n')

def stats_of_a_list(arr):
    rms = Statistics.root_mean_square(arr)
    mean = Statistics.arithmetic_mean(arr)
    gm = Statistics.geometric_mean(arr)
    hm = Statistics.harmonic_mean(arr)
    cubic_mean = Statistics.generalized_mean(arr, 3)
    quartic_mean = Statistics.generalized_mean(arr, 4)
    quintic_mean = Statistics.generalized_mean(arr, 5)
    median = Statistics.median(arr)
    upper_quartile_median = Statistics.upper_quartile_median(arr)
    lower_quartile_median = Statistics.lower_quartile_median(arr)
    range = Statistics.range(arr)
    mid_range = Statistics.mid_range(arr)
    standard_deviation = Statistics.standard_deviation(arr)
    cubic_standard_deviation = Statistics.generalized_standard_deviation(arr, 3)
    quartic_standard_deviation = Statistics.generalized_standard_deviation(arr, 4)
    quintic_standard_deviation = Statistics.generalized_standard_deviation(arr, 5)

    return [rms, mean, gm, hm, cubic_mean, quartic_mean, quintic_mean, median, upper_quartile_median, lower_quartile_median, range, mid_range, standard_deviation, cubic_standard_deviation
            , quartic_standard_deviation, quintic_standard_deviation]


stats_and_stock_prices = []
#I only need 'snippet', 'lead_paragraph'
for article_stock in articles_and_stock_price:
    articles = article_stock[0]
    stock_prices = article_stock[1]
    predictions = []
    snippets = []
    lead_paragraphs = []
    for article in articles:
        snippet = article['snippet']
        lead_paragraph = article['lead_paragraph']
        snippet_sentiment = predict(snippet)
        lead_paragraph_sentiment = predict(lead_paragraph)
        # put through the sentiment model
        #sentiment_and_stock_prices.append([[predict(snippet), predict(lead_paragraph)], stock_prices])
        predictions.append([snippet_sentiment, lead_paragraph_sentiment])


    snippets_stats = stats_of_a_list(snippets) #make sure to use the list for both of these
    lead_paragraph_stats = stats_of_a_list(lead_paragraphs)
    stats_and_stock_prices.append([[snippets_stats, lead_paragraph_stats], stock_prices])


#now have the articles with the stats of the sentiment

days_in_a_row = 7 #we are using the data from a week to try and predict if the market will go up or down
average_of_n_days = 4 #take the average price of the next n days (n = 4 here) which will be used to predict if the market will go up or down

csv = f'{stock} {days_in_a_row} days in a row with the next {average_of_n_days} days.csv'

if csv in os.listdir() or csv in os.listdir('../ML_Prediction/'):
		raise Exception("The csv of data already exists")
f = open(csv, 'a')

# headers for the csv:
# Snippet Dn, Lead Paragraph Dn, Open Dn, High Dn, Low Dn, Low Dn, Close Dn, Adj Close Dn, Volume Dn where n is the day #
#after those there will be more headers of "average of the next {average_of_n_days}", comparison at the very end
#8 commas

for day in range(1, days_in_a_row + 1): #the +1 is since range isn't inclusive
    f.write(f'Snippet D{day}\t')
    f.write(f'Lead Paragraph D{day}\t')
    f.write(f'Open D{day}\t')
    f.write(f'High D{day}\t')
    f.write(f'Low D{day}\t')
    f.write(f'Close D{day}\t')
    f.write(f'Volume D{day}\t')

f.write(f'average of the next {average_of_n_days}\t')
f.write('comparison\n')

values = []
future_values = []

for i in range(days_in_a_row):
    values.append([stats_and_stock_prices[i][0][0],
                  stats_and_stock_prices[i][0][1],
                  stats_and_stock_prices[i][1][0],
                  stats_and_stock_prices[i][1][1],
                  stats_and_stock_prices[i][1][2],
                  stats_and_stock_prices[i][1][3],
                  stats_and_stock_prices[i][1][4],
                  stats_and_stock_prices[i][1][5],
                  stats_and_stock_prices[i][1][6],
                  ])

for i in range(days_in_a_row, days_in_a_row + average_of_n_days):
    future_values.append([stats_and_stock_prices[i][0][0],
                  stats_and_stock_prices[i][0][1],
                  stats_and_stock_prices[i][1][0],
                  stats_and_stock_prices[i][1][1],
                  stats_and_stock_prices[i][1][2],
                  stats_and_stock_prices[i][1][3],
                  stats_and_stock_prices[i][1][4],
                  stats_and_stock_prices[i][1][5],
                  stats_and_stock_prices[i][1][6],
                  ])

print(values)




























    #making some whitespace with this comment
