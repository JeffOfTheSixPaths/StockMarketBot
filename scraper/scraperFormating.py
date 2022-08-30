import Scrap
import pandas as pd
from datetime import date
import json
import os

today = date.today()

go_back = 2 #years, can do a version for months and days if you need, but then it can get a little weird with over/underflows

#subtracts go_back from the current year
init_date = date(today.year - go_back, today.month, today.day)

print(f'today\'s date is {today}')
print(f'I\'m going back to {init_date}')

list_of_dates = pd.date_range(start = init_date, end = today, freq = 'D').tolist()
#removes the 00:00:00 timestamp at the end of the thing
for i in range(len(list_of_dates)):
    list_of_dates[i] = str(list_of_dates[i].date())


for d in list_of_dates[0:5]:
    print(d)

list_of_months = pd.date_range(start = init_date, end = today, freq = 'M').tolist()
#removing the 00:00:00 timestamp from the end of list_of_months
months_to_articles = { #the articles that appear in each month in the form of a dictionary, example:
                        #'2021-8': [list of articles], etc.
                    }

api_key = ""
for month in list_of_months[0:1]:
    months_to_articles[f'{month.year}-{month.month}'] = Scrap.get_nyt(f'/{month.year}/{month.month}', api_key)# need to replace all NaN with None
    # a little pseudocode -> months_to_articles[month] = get_nyt(month, api_key) 

days_to_articles = {# same thing as months_to_articles but with days.
                    #ex, 2021-12-31 : <some pd dataframe>
                    }       

for month in months_to_articles:
    dataframe = months_to_articles[month]
    '''
    replace NaN with None
    sort by 'pub_date'
    splice pub date into just the date without the timestamp (might need to switch this and the one
    above cause pd might not be able to sort timestamps with a T in them)

    problem: the articles are sorted into numbers, so I have to go through and sperately combine all of them

    append to days_to_articles with the solution above
    save days_to_articles to a json

    I could just sort them into seperate articles
    do computers have a divisible number of bytes?
    should only do this research for a specific set of stocks
    
    articles = {}
    for article in sorted_dataframe['pub_date']: #the key is wrong, the month needs to come before 'pub_date'
        for key in sorted_dataframe:
            article_data = #concat the keys for the article
        articles[sorted_dataframe['pub_date'][article]] = article_data
        ^ makes a key for a date if the date isn't already there and then the comment explains itself

    '''








