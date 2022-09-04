import APIs
import pandas as pd
from datetime import date
import json
import os
import time

today = date.today()

go_back = 2 #years, can do a version for months and days if you need, but then it can get a little weird with over/underflows

#subtracts go_back from the current year
init_date = date(today.year - go_back, today.month, today.day)

print(f'today\'s date is {today}')
print(f'I\'m going back to {init_date}')

list_of_dates = pd.date_range(start = init_date, end = today, freq = 'D').tolist()

#removes the 00:00:00 timestamp at the end of the date
for i in range(len(list_of_dates)):
    list_of_dates[i] = str(list_of_dates[i].date())

list_of_months = pd.date_range(start = init_date, end = today, freq = 'M').tolist()
#removing the 00:00:00 timestamp from the end of list_of_months
months_to_articles = { #the articles that appear in each month in the form of a dictionary, example:
                        #'2021-8': [list of articles], etc.
                    }

api_key = ""
for month in list_of_months[0:1]: #
    months_to_articles[f'{month.year}-{month.month}'] = APIs.get_nyt(f'/{month.year}/{month.month}', api_key)# need to replace all NaN with None
    # ^ the dataframe is normalized

    # a little pseudocode -> months_to_articles[month] = get_nyt(month, api_key)
    #each element is a dataframe in months_to_articles

days_to_articles = {# same thing as months_to_articles but with days.
                    #ex, 2021-12-31 : <some pd dataframe>
                    }
NaN = None #supposed to stop the program from throwing error when it encounters NaN in the response from the apis

'''
for the for loop below:

we assign the current month to a dataframe and then sort that dataframe by the publish date.
we then go through each publish date and remove the T00:00:00 from them (for example 2022-08-01T00:00:00 to 2022-08-01).

we now go through each article individually and add their keys to a dictionary article_dictionary which will later be used to be appended into days_to_articles.
continuing to go through each individual article, we add the publish date to the dictionary days_to_articles. if the publish date is not there it adds an empty
list as its value. the article (article_dictionary) is then appended to that list.
^is continued for all the articles and then saved as a json.
'''
for month in months_to_articles:
    dataframe = months_to_articles[month]

    sorted_dataframe = dataframe.sort_values(by='pub_date')

    for index, element in enumerate(sorted_dataframe['pub_date']):
        # index is the id of the article it's working on
        sorted_dataframe['pub_date'][index] = str(element).split('T')[0] #get just the YY-MM-DD in the timestamp

        article_dictionary = {}
        for key in sorted_dataframe:
            article_dictionary[key] = str(sorted_dataframe[key][index])
        #article_dictionary = pd.json_normalize(article_dictionary)
        #article_dictionary.columns = sorted_dataframe.columns

        try:
            type(days_to_articles[str(article_dictionary['pub_date'])]) == type([]) #if the element of a key is a list
        except:
            days_to_articles[str(article_dictionary['pub_date'])] = []

        days_to_articles[str(article_dictionary['pub_date'])].append(article_dictionary) #the date of the article is the key, and the article gets appended to a array of dataframes about the days'
        # articles

        #^ is a dictionary, so save it with json

#json save days_to_articles
f = open("days_to_articles.json", 'w')
json.dump(days_to_articles, f)
f.close()

'''

days_to_articles = {
                'date': [article_dictionary1, article_dictionary2, etc.],
                'another date': [the articles on this date now as a dictionary, another article on this date as a dictionary]
                all the articles should be added to their respective dates, although the json is not easily human readable.

}

'''
