from base64 import standard_b64decode
from ipaddress import AddressValueError
from multiprocessing.sharedctypes import Value
from zoneinfo import available_timezones
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow.keras
import pandas as pd
import numpy as np
import json
import yfinance as yf
import os
import sys
import Formating_the_APIs as fta

#this just imports sentimentAnalysis.py in this directory
import_path= '../ML_Prediction/Sentiment'
sys.path.insert(1, import_path)

import sentimentAnalysis
import Statistics #not to be confused with the statistics module that's preinstalled, this is from Statistics.py
#making the data from the other files into a form more usuable for AI.

print(os.getcwd())
config = open('../config.json' , 'r')
config = json.load(config)

def make_ticker(ticker: str): #this might be useless
    return yf.Ticker(ticker)

#the dates is represented as yy-mm-dd
#the period might need to be changed depending on the length of time we go back
period = str(config['go back how many years'])+'y' #  ex. config['go back how many years] = 1 then period = 1 + 'y' = '1y'

def get_stock_data(tickers: str, period = period, interval = '1d'): #gets the stock prices for the previous period of time
    return yf.download(tickers, period = period, interval = interval)


def make_d2a(): #make d2a (days 2 articles)
    #this json has each of the days and the articles made on each of those days
    print('Making days_to_articles.json')
    fta.get_d2a() #this file makes the days_to_articles.json which has the days with the articles published on those days as a dictionary
    print("Finished making days_to_articles.json")

def load_d2a():
    data = open('days_to_articles.json', 'r')

    print("loading the data from the json")
    articles = json.load(data)
    print('loaded the data from the json successfully')
    return articles

#checks for reusing an existing days_to_articles.json
reuse_json = config["reuse days_to_articles.json?"]
if reuse_json == "y" and not ('days_to_articles.json' in os.listdir()):
    print("reusing the days_to_articles.json is on, but couldn't find it\n making it again")
    make_d2a()
    articles = load_d2a()
elif reuse_json == "y" and ('days_to_articles.json' in os.listdir()):
    print("found days_to_articles.json")
    articles = load_d2a()
elif reuse_json == 'n':
    print("chose not the reuse days_to_articles.json\n making it again")
    make_d2a()
    articles = load_d2a()
elif not ( reuse_json == 'n' or 'y'):
    raise ValueError(" reuse days_to_articles.json is not 'y' or 'n' ")


print(f"getting stock prices for the past {config['go back how many years']} years")
stock = config['stock'] #change the stock to track here
stock_data = get_stock_data(stock) #gets the stock prices for the past two years
print("got the stock prices")

#this part is making a dictionary of dates with their values being the stock prices for that day
list_of_dates = fta.list_of_dates
dates_of_the_stock = {} #list of all the dates that the stock was traded for as a dictionary
stock_data['Date'] = stock_data.index #just adds the Date index as a coloumn

#making a dictionary with the date as the key and stock data on that day as the values
#this gives a date to stock price conversion
for index, date in enumerate(stock_data['Date']): #The date and the corresponding stock's prices for that day
    dates_of_the_stock[str(date).split(' ')[0]] = [stock_data['Open'][index], stock_data['High'][index], stock_data['Low'][index], stock_data['Close'][index], stock_data['Adj Close'][index], stock_data['Volume'][index]]

#print(list(dates_of_the_stock.keys())[0]) # earliest date

#list_of_stock_dates is a list of the dates that the stock has a price on
list_of_stock_dates = list(dates_of_the_stock.keys())# each dates is in the format yy-mm-dd 00:00:00
first_stock_date = list_of_stock_dates[0] #the first date in list_of_stock_dates which is the earliest date that exists
#nyt not used for any specific reason besides that it has almost every single date

print(f'the first date recognized is {first_stock_date}')
first_stock_date = str(first_stock_date).split(' ')[0] #removes the " 00:00:00" from the timestamp



#[articles,stock prices]
articles_and_stock_price = [[articles[first_stock_date], dates_of_the_stock[first_stock_date]]] #this is a 2d array

#finds which date in the list of dates the stocks prices start on
starting_date_index = 0
for index, date in enumerate(list_of_dates):
    if date == first_stock_date:
        starting_date_index = index
        break

#print(starting_date_index)

for index, date in enumerate(list_of_dates[starting_date_index+1:]):
    if date in list_of_stock_dates:
        if date not in articles.keys():
            print(f'{date} is not in articles')
        elif date not in dates_of_the_stock.keys():
            print(f'{date} in not in dates_of_the_stock')
        articles_and_stock_price.append([articles[date], dates_of_the_stock[date]])

    else:
        #if the date is not in the list of the stocks prices, then this gets the last stock price and assigns it to the current date
        last_values = articles_and_stock_price[-1] #gets last stock price
        last_stock_prices = last_values[1] #the second index is the array of stock prices
        articles_and_stock_price.append([articles[date], last_stock_prices]) #assigns the date the last known stock price



#now for the ai
def download_sentiment_data():
    print("Starting on making the AI")
    print("Downloading necessary files from a google drive")
    print("if the files are missing, open an issue")
    folder_id = '1MifzRW3qeJXdPfVdb7xz8Q6ITjtG6u9A'
    os.system(f'gdown --folder {folder_id}') #downloads the folder called "sentiment_model_weights" from a google drive

redownload_sentiment_data = config["redownload sentiment analysis model data?"]

if redownload_sentiment_data == 'n':
    if 'sentiment_model_weights' in os.listdir():
        print("not redownloading sentiment analysis data")
    else:
        print('downloading sentiment analysis data')
        download_sentiment_data()
elif redownload_sentiment_data == "y":
    print('redownloading sentiment analysis data')
    download_sentiment_data()
elif not (redownload_sentiment_data == 'y' or redownload_sentiment_data == 'n'):
    raise ValueError("redownload_sentiment_data is not 'y' or 'n' ")

use_s140 = False # Putting this to True will break the code, but this will be kept in here in case I use the s140 dataset in my sentiment analysis model sometime in the future

weights_path = 'sentiment_model_weights/cp.cpkt'
weights_dir = os.path.dirname(weights_path)
latest = tf.train.latest_checkpoint(weights_dir)

train_text = 'sentiment_model_weights/sentiment_text.csv'
train_sentiment = 'sentiment_model_weights/sentiment_sentiment.npy'
if use_s140:
    s140_text = 'sentiment_model_weights/sentiment140_text.csv'
    s140_sentiment = 'sentiment_model_weights/sentiment140_sentiment.npy'

# x_text holds the sentences and x_sentiment has the sentiment to x_text's sentences
print("loading the data")
train_text = pd.read_csv(train_text)
train_sentiment = np.load(train_sentiment)
train_text.pop('Unnamed: 0')
if use_s140:
    s140_text = pd.read_csv(s140_text)
    s140_sentiment = np.load(s140_sentiment)
    s140_text.pop('Unnamed: 0')
print("loaded the data")

print("converting data to a usable form for the ai")
train_text = tf.convert_to_tensor(train_text)

if use_s140:
    s140_tensor = tf.convert_to_tensor(s140_text)
    train_text = tf.concat([train_text,s140_tensor], 0)


train_sentiment = tf.convert_to_tensor(train_sentiment)
train_dataset = tf.data.Dataset.from_tensor_slices((train_text,train_sentiment))
print("converted")

print('creating the model and encoding the data')
model = sentimentAnalysis.make_sentiment_model(train_dataset)
print('loading weights')
model.load_weights(latest)
print('loaded weights')
def predict(sentence: str):
    return model.predict(np.array([sentence]), verbose = 0)

test_sentences = [
                    'I love this',
                    'I hate this',
                    'Gamers are oppressed',
                    'I really love dogs',
                    'I really hate dogs',
                    'I fucking love dogs', #using the f-word here as it can be used with different connotations in different sentences
                    'I fucking hate dogs', #it can also be used as a stronger form of "really"
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


#since each day has a variable number of articles, it is necessary to create some way to reduce it to some constant number
#my approach is to generate many statistics about the set of articles on each day
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
    #article_stock is a 2d list of [articles, stock_prices]
    articles = article_stock[0]
    stock_prices = article_stock[1]
    snippets = []
    lead_paragraphs = []

    '''
    for article in articles:
        get the snippet and the lead paragraph
        get the sentiment of the snippet and lead paragraph
        then put the sentiment of the snippet into the array snippets
        and put the sentiment of the lead paragraph into the array lead_paragraphs

        we put the sentiment into snippets and lead_paragraphs so that we can use stats_of_a_list() on both of them easily
    '''

    for article in articles['nyt']:
        snippet = article['snippet']
        lead_paragraph = article['lead_paragraph']
        try:
            snippet_sentiment = predict(snippet)
            lead_paragraph_sentiment = predict(lead_paragraph)
        except:
            print(article['pub_date'])
        else:
            # put through the sentiment model
            snippets.append(snippet_sentiment)
            lead_paragraphs.append(lead_paragraph_sentiment)
    try:
        for article in articles['theguardian']:
            snippet = article['webTitle']
            try:
                snippet_sentiment = predict(snippet)
            except:
                print(article['webPublicationDate'])
            
            else:
                snippets.append(snippet_sentiment)
    except:
        print('couldnt guardian')

    #creates the statistics of the snippets and lead paragraphs
    snippets_stats = stats_of_a_list(snippets) #make sure to use the list for both of these
    lead_paragraph_stats = stats_of_a_list(lead_paragraphs)

    #each day has its own stats and lead paragraph stats as well as stock prices
    stats_and_stock_prices.append([[snippets_stats, lead_paragraph_stats], stock_prices])


#now have the articles with the stats of the sentiment

days_in_a_row = config["number of days in a row to analyze"] #we are using the data from a week to try and predict if the market will go up or down
num_of_future_days_to_take_average_of = config["number of days in the future to take the average of"]  #take the average price of the next n days (n = 4 here) which will be used to predict if the market will go up or down

# headers for the csv:
# Snippet Dn, Lead Paragraph Dn, Open Dn, High Dn, Low Dn, Low Dn, Close Dn, Adj Close Dn, Volume Dn where n is the day #
#after those there will be more headers of "average of the next {num_of_future_days_to_take_average_of}", comparison at the very end
#8 commas

headers = [] #headers of the csv
headers.append('pop this') #used to initilize the dataframe with some junk values that will be removed later
for day in range(1, days_in_a_row + 1): #the +1 is since range isn't inclusive
    headers.append(f'Snippet D{day}')
    headers.append(f'Lead Paragraph D{day}')
    headers.append(f'Prices D{day}')

headers.append(f'average of the next {num_of_future_days_to_take_average_of}')
headers.append('comparison')
print(headers)

values = []
future_values = []

def convert_data(arr): #converts data into a more readable form
    for a,k in enumerate(arr):
            k = k.tolist()[0][0] #each element k is in the form [[k]] even though there's only a single element
            if k == np.inf:
                k = 99
            elif k == np.nan:
                k = 0
            arr[a] = k # .tolist()[0][0]
    return arr

for i in range(days_in_a_row):
    # [i][0] is part of the [snippet, lead paragraph]
    # [i][1] is part of the stock prices
    arr = []
    for j in stats_and_stock_prices[i][0]:
        convert_data(j)
        arr.append(j)

    arr.append(stats_and_stock_prices[i][1])

    values.append(arr)

#initial future_values[]
for i in range(days_in_a_row, days_in_a_row + num_of_future_days_to_take_average_of):
    future_values.append(stats_and_stock_prices[i][1])


compare_stock_type = config['stock compare type']
#make the future values only have the values of the stock type of your choosing
for i,e  in enumerate(future_values):
    future_values[i] = e[compare_stock_type]

#initial values of the average and comparison
average_of_next_days = Statistics.arithmetic_mean(future_values) #just the average of future_values
comparison = average_of_next_days > values[-1][1][compare_stock_type] #comparison compares if the average of the next few days is greater than the last price the values ended on
#as of writing, it compares the average to values[-1][1][0], which is the Open price, changing compare_stock_type to another number (e.i compare_stock_type = 1) would make it a different price
#the index are as follows from 0 - 6
#Open, High, Low, Low, Close, Adj Close, Volume

dict_for_formating = {}
junk_values = [1,2]
df_of_all_data = pd.DataFrame({0:[junk_values]}) #allows the dataframe to store the array in one cell
for day in values:
    print(day[2])
    df = pd.DataFrame({
        0:[day[0]], #should be the snippet sentiment
        1:[day[1]], #should be the lead paragraph sentiment
        2:[day[2]] #should be all the different prices
    })
    df_of_all_data = pd.concat([df_of_all_data, df], axis = 1)

df_of_all_data.head()

#future values do not need to be a dataframe since the ai does not feed on future_values but instead *could* need the average
a = pd.DataFrame([average_of_next_days, comparison]).transpose()
dataframe_of_average_and_comparison = a
#print(future_values_dataframe.head())
print(dataframe_of_average_and_comparison.head())

df_of_all_data = pd.concat([df_of_all_data, dataframe_of_average_and_comparison], axis = 1)
df_of_all_data.columns = headers
df_of_all_data.pop('pop this')


def convert_data(arr): #converts data into a more readable form
    for a,k in enumerate(arr):
        try:
            k = k.tolist()[0][0] #each element k is in the form [[k]] even though there's only a single element
        except:
            pass
        if k == np.inf:
            k = 99
        elif k == np.nan or k == None:
            k = 0
        arr[a] = k # .tolist()[0][0]
    return arr


for i in range(days_in_a_row, len(stats_and_stock_prices) - num_of_future_days_to_take_average_of - 1 ):
    #adjusting the array of values
    '''
    for both values and future_values,
        get rid of the oldest day's data
        append to the array the next avalible day's data
    '''
    values.pop(0) #get rid of the oldest day

    arr = []
    for j in stats_and_stock_prices[i][0]:
        convert_data(j)
        arr.append(j)

    arr.append(stats_and_stock_prices[i][1]) #adds the stock prices for that day

    values.append(arr) #add the new day, which is future_values[0] before we pop it in the next line

    #adjusting the values of the future days
    future = i + num_of_future_days_to_take_average_of
    future_values.pop(0)

    future_values.append(stats_and_stock_prices[future][1][compare_stock_type])

    average_of_next_days = Statistics.arithmetic_mean(future_values) #just the average of future_values
    comparison = average_of_next_days > values[-1][1][compare_stock_type] #comparison compares if the average of the next few days is greater than the last price the values ended on

    dict_for_formating = {}
    df_of_sentiment_and_prices = pd.DataFrame({0:[junk_values]}) #allows the dataframe to store the array in one cell

    #converts the arrays into a form that can be put into dataframes, this will have to be changed manually if more data is going to be put in
    for day in values:
        df = pd.DataFrame({
            0:[day[0]], #should be the snippet sentiment
            1:[day[1]], #should be the lead paragraph sentiment
            2:[day[2]] #should be all the different prices
        })
        df_of_sentiment_and_prices = pd.concat([df_of_sentiment_and_prices, df], axis = 1)

    dataframe_of_average_and_comparison = pd.DataFrame([average_of_next_days, comparison]).transpose()
    new_data = pd.concat([df_of_sentiment_and_prices, dataframe_of_average_and_comparison], axis = 1)
    new_data.columns = headers
    new_data.pop('pop this') #removes the column of junk data before concating the rest of the data
    df_of_all_data = pd.concat([df_of_all_data, new_data], axis = 0) #appends new data to the a new row at the end of the dataframe
    del new_data


df_of_all_data.to_csv(config['csv name'], sep = '\t') #change this later to something more descriptive
