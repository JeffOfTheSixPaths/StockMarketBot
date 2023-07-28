from base64 import standard_b64decode
from ipaddress import AddressValueError
from multiprocessing.sharedctypes import Value
from zoneinfo import available_timezones
#
import pandas as pd
import numpy as np
import json
import yfinance as yf
import os
import sys
import Formating_the_APIs as fta

#this just imports sentimentAnalysis.py in this directory
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


#list_of_stock_dates is a list of the dates that the stock has a price on
list_of_stock_dates = list(dates_of_the_stock.keys())# each dates is in the format yy-mm-dd 00:00:00
first_stock_date = list_of_stock_dates[0] #the first date in list_of_stock_dates which is the earliest date that exists
#nyt not used for any specific reason besides that it has almost every single date

print(f'the first date recognized is {first_stock_date}')
first_stock_date = str(first_stock_date).split(' ')[0] #removes the " 00:00:00" from the timestamp



d2_articles_stocks = { # name is dates to articles stocks
    first_stock_date: [articles[first_stock_date], dates_of_the_stock[first_stock_date]]
 }

# each element of this is [articles, stock price], articles is a dict and dates_of_the_stock is an array
articles_and_stock_price = [[articles[first_stock_date], dates_of_the_stock[first_stock_date]]] #this is a 3d array

#finds which date in the list of dates the stocks prices start on
starting_date_index = 0
for index, date in enumerate(list_of_dates):
    if date == first_stock_date:
        starting_date_index = index
        break


# many stocks have missing dates in them, this is filling in the gaps
# if there's a missing value in the stock prices, it just fills it in with the last known stock date
for index, date in enumerate(list_of_dates[starting_date_index+1:]):
    # checks if the stock has the date
    if date in list_of_stock_dates:
        if date not in articles.keys():
            print(f'{date} is not in articles')
        elif date not in dates_of_the_stock.keys():
            print(f'{date} in not in dates_of_the_stock')
        
        d2_articles_stocks[date] = [articles[date], dates_of_the_stock[date]]
        articles_and_stock_price.append([articles[date], dates_of_the_stock[date]])

    #if it doens't have the date it goes back and fills in the value with the last known stock price
    else:
        #if the date is not in the list of the stocks prices, then this gets the last stock price and assigns it to the current date
        last_values = articles_and_stock_price[-1] #gets last stock price
        last_stock_prices = last_values[1] #the second index is the array of stock prices
        d2_articles_stocks[date] = [articles[date], last_stock_prices]
        articles_and_stock_price.append([articles[date], last_stock_prices]) #assigns the date the last known stock price

from datetime import datetime
sorted_dates = list(d2_articles_stocks.keys()) # getting the list of dates
sorted_dates.sort( key = lambda date: datetime.strptime(date, '%Y-%m-%d'))
data = []
days_in_row = config["number of days in a row to analyze"] #we are using the data from a week to try and predict if the market will go up or down
num_future = config["number of days in the future to take the average of"]  #take the average price of the next n days (n = 4 here) which will be used to predict if the market will go up or down"

for i in range(0, len(sorted_dates) - ( num_future + days_in_row) ):
    current_dates = sorted_dates[i: i + days_in_row]
    future_dates = sorted_dates [i + days_in_row:     i + days_in_row + num_future]

    #map_func = lambda date: d2_articles_stocks[date]

    #current_dates = list( map( map_func, current_dates ) )  IT'S SO SLOW!!!!
    #future_dates = list( map( map_func, future_dates ) )
    #result = current_dates + future_dates
    #print(result)
    data.append(current_dates + future_dates)

data = pd.DataFrame(data)
#data
'''

             0	         1	         2	         3	         4	         5	         6	         7	        8	         9	        10
0	2022-07-28	2022-07-29	2022-07-30	2022-07-31	2022-08-01	2022-08-02	2022-08-03	2022-08-04	2022-08-05	2022-08-06	2022-08-07
1	2022-07-29	2022-07-30	2022-07-31	2022-08-01	2022-08-02	2022-08-03	2022-08-04	2022-08-05	2022-08-06	2022-08-07	2022-08-08
2	2022-07-30	2022-07-31	2022-08-01	2022-08-02	2022-08-03	2022-08-04	2022-08-05	2022-08-06	2022-08-07	2022-08-08	2022-08-09
3	2022-07-31	2022-08-01	2022-08-02	2022-08-03	2022-08-04	2022-08-05	2022-08-06	2022-08-07	2022-08-08	2022-08-09	2022-08-10
4	2022-08-01	2022-08-02	2022-08-03	2022-08-04	2022-08-05	2022-08-06	2022-08-07	2022-08-08	2022-08-09	2022-08-10	2022-08-11

'''

def get_ap(df: pd.DataFrame): # a mapping function to get ap (Articles and Prices) from the dates in the row
    df_cp  = df
    for col in df.columns:
        tmp_df = df[col]
        nyt = []
        guard = []
        prc = [] # prices
        for date in tmp_df:
            #print(type(d2_articles_stocks))
            d = d2_articles_stocks[date]
            #print(d[0].keys())
            nyt.append(d[0].get('nyt', []))
            guard.append(d[0].get('theguardian', []))
            prc.append(d[1])
        
        df_cp['nyt' + str(col)] = pd.DataFrame([nyt]).transpose()
        df_cp['guard' + str(col)] = pd.DataFrame([guard]).transpose()
        df_cp['prices' + str(col)] = pd.DataFrame([prc]).transpose()
    return df_cp

data = get_ap(data)

print(data.columns)
''' ^output
Index([         0,          1,          2,          3,          4,          5,
                6,          7,          8,          9,         10,     'nyt0',
         'guard0',  'prices0',     'nyt1',   'guard1',  'prices1',     'nyt2',
         'guard2',  'prices2',     'nyt3',   'guard3',  'prices3',     'nyt4',
         'guard4',  'prices4',     'nyt5',   'guard5',  'prices5',     'nyt6',
         'guard6',  'prices6',     'nyt7',   'guard7',  'prices7',     'nyt8',
         'guard8',  'prices8',     'nyt9',   'guard9',  'prices9',    'nyt10',
        'guard10', 'prices10'],
      dtype='object')
'''
print('\n\n\n')
print(data.head())
'''
            0           1           2           3  ...                                            prices9                                              nyt10 guard10                                           prices10
0  2022-07-28  2022-07-29  2022-07-30  2022-07-31  ...  [279.1499938964844, 283.6499938964844, 278.679...  [{'abstract': 'Quotation of the Day for Sunday...      []  [279.1499938964844, 283.6499938964844, 278.679...
1  2022-07-29  2022-07-30  2022-07-31  2022-08-01  ...  [279.1499938964844, 283.6499938964844, 278.679...  [{'abstract': 'The Mets won four of five games...      []  [284.04998779296875, 285.9200134277344, 279.32...
2  2022-07-30  2022-07-31  2022-08-01  2022-08-02  ...  [284.04998779296875, 285.9200134277344, 279.32...  [{'abstract': 'The F.B.I.’s search of Mar-a-La...      []  [279.6400146484375, 283.0799865722656, 277.609...
3  2022-07-31  2022-08-01  2022-08-02  2022-08-03  ...  [279.6400146484375, 283.0799865722656, 277.609...  [{'abstract': 'Justice Department officials we...      []  [288.1700134277344, 289.80999755859375, 286.94...
4  2022-08-01  2022-08-02  2022-08-03  2022-08-04  ...  [288.1700134277344, 289.80999755859375, 286.94...  [{'abstract': 'David W. Tuffs’s puzzle is posi...      []  [290.8500061035156, 291.2099914550781, 286.510...

[5 rows x 44 columns]
'''

data.to_csv(config['csv name'], sep = '†') # using an extremely rare sep so that it doesn't conflict with the data
