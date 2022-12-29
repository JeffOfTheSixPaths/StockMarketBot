import APIs
import pandas as pd
from datetime import date
import json
import os
import time
import threading


nyt_d2a = []
guard_d2a = []

config = open('../config.json' , 'r')
config = json.load(config)

today = date.today()

go_back = config['go back how many years'] #years, can do a version for months and days if you need, but then it can get a little weird with over/underflows

#subtracts go_back from the current year
init_date = date(today.year - go_back, today.month, today.day)

print(f'today\'s date is {today}')
print(f'I\'m going back to {init_date}')

list_of_dates = pd.date_range(start = init_date, end = today, freq = 'D').tolist()

#removes the 00:00:00 timestamp at the end of the date
for i in range(len(list_of_dates)):
    list_of_dates[i] = str(list_of_dates[i].date())

#due to how the pd.date_range works, the init_date needs to be changed to the beginning of the month since the date_range
#pretty much says that the month only counts if it has a beginning within 2 years
init_date = f'{init_date.year}-{init_date.month}-01'
list_of_months = pd.date_range(start = init_date, end = today, freq = 'MS', inclusive = 'both')

list_of_months = list_of_months.tolist()

#months_to_articles[some_month] == the articles in that month
months_to_articles = { #the articles that appear in each month in the form of a dictionary, example:
                        #'2021-8': [list of articles], etc.
                    }

nyt_api_key = config['nyt archive api key']
def make_d2a_nyt():
    if nyt_api_key == "":
        raise Exception("nyt archive api key is empty")

    """
    this function gets every month's articles and puts them in a dictionary with the label being the month date. 
    It then goes through every article and puts them in a dictionary with the key being the date
    """
    for month in list_of_months: #
        print(f'requesting nyt with the date /{month.year}/{month.month}' )
        months_to_articles[f'{month.year}-{month.month}'] = APIs.get_nyt(f'/{month.year}/{month.month}', nyt_api_key)# need to replace all NaN with None
        # ^ the dataframe is normalized

        # a little pseudocode -> months_to_articles[month] = get_nyt(month, api_key)
        #each element is a dataframe/dictionary in months_to_articles

    days_to_articles = {# same thing as months_to_articles but with days.
                        #ex, 2021-12-31 : <array of dictionaries>
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

            #going through each article
            article_dictionary = {}
            for key in sorted_dataframe:
                article_dictionary[key] = str(sorted_dataframe[key][index])

            try:
                type(days_to_articles[str(article_dictionary['pub_date'])]) == type([]) #if the element of a key is a list
            except:
                days_to_articles[str(article_dictionary['pub_date'])] = []

            days_to_articles[str(article_dictionary['pub_date'])].append(article_dictionary) #the date of the article is the key, and the article gets appended to a array of dataframes about the days'
            # articles

            #^ is a dictionary, so save it with json
    nyt_d2a.append(days_to_articles)
    return days_to_articles
    #json save days_to_articles
    f = open("days_to_articles.json", 'w')
    json.dump(days_to_articles, f)
    f.close()

guard_api_key = config["theguardian api key"]
def make_d2a_theguardian():
    if guard_api_key == "":
        raise ValueError("guardian api key is empty")
        
    phrases = config["phrases for filtering"]

    from_date = init_date # this is the current date minus how many years the config file specifies
    
    guardian = APIs.get_guardian(from_date=from_date, phrases=phrases, api_key=guard_api_key)
    
    days_to_articles = {}
    for i in guardian:
        try:
            days_to_articles[i['webPublicationDate']].append(i)
        except:
            print(i)
            days_to_articles[i['webPublicationDate']] = [i]
    guard_d2a.append(days_to_articles)
    return days_to_articles
    

'''

days_to_articles = {
                'date': [article_dictionary1, article_dictionary2, etc.],
                'another date': [the articles on this date now as a dictionary, another article on this date as a dictionary]
                all the articles should be added to their respective dates, although the json is not easily human readable.

}

'''

def get_d2a():
    #making multiple threads so that we don't have to wait for all of the requests to be over before we can start the next one
    nyt_thread = threading.Thread(target = make_d2a_nyt)
    guard_thread = threading.Thread(target = make_d2a_theguardian)

    print('starting the threads')
    nyt_thread.start()
    guard_thread.start()

    nyt_thread.join()
    guard_thread.join()
    print('threads are finished')
    #print(f'the number of threads: {threading.active_count()}') this prints 1

    see_nyt = False
    if see_nyt:
        f = open("nyt_d2a.json",'w')
        json.dump(nyt_d2a[0],f) #it's nyt_d2a[0] because the d2a is appended to an empty array in order to have it returned
        f.close()

    see_guardian = False
    if see_guardian:
        f = open("guardian_d2a.json", 'w')
        json.dump(guard_d2a[0], f) #it's guard_d2a[0] because the d2a is appended to an empty array in order to have it returned
        f.close()
    
    
    

    days_to_articles = {}
    for key, element in nyt_d2a[0].items(): #it's nyt_d2a[0] because the d2a is appended to an empty array in order to have it returned
        days_to_articles[key] = {}
        days_to_articles[key]['nyt'] = element
    

    for key, element in guard_d2a[0].items(): #it's guard_d2a[0] because the d2a is appended to an empty array in order to have it returned
        try:
            days_to_articles[key]['theguardian'] = element
        except:
            days_to_articles[key] = {}
            days_to_articles[key]['theguardian'] = element

    
    f = open("days_to_articles.json", 'w')
    json.dump(days_to_articles, f, indent=2)
    f.close()
    return days_to_articles

if __name__ == '__main__':
    get_d2a()