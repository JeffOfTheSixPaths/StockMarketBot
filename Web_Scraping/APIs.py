## python webscraper
import requests
from bs4 import BeautifulSoup
import json
import yfinance as yf
import pandas as pd
import time
## lxml parser downloaded

def rmAngles(string_with_tags):
	# removes the tags by finding all the < and > symbols and removing everything between them

	string_with_tags = str(string_with_tags)

	## less is < "less than" and greater is > "greater than"
	less_size = len(string_with_tags.split("<"))
	greater_size = len(string_with_tags.split(">"))


	## less == greater since it's parsing HTML tags
	if less_size == greater_size:
		for i in range(int(less_size)-1):
			less = string_with_tags.find("<")
			greater = string_with_tags.find(">")
			string_with_tags = string_with_tags[:less] + string_with_tags[greater+1:]
			#string=string[:index] + string[index+1:] removes the character at [index], but not at [index+1]
	else:
		print("Error: spare \"<\" or \">\" character in: \n" + string_with_tags) # if there was a spare, the parsing algorithm would break
	return string_with_tags ##don't want to do string = string_with_tags somewhere just for return string

def get_rec(ticker, amount): #get recommendations
	diction = ticker.recommendations
	recom = []
	for i in range(amount):
		# (firm, to grade, action)
		recom.append([diction["Firm"][(-1*amount)+i],diction["To Grade"][(-1*amount)+i],diction["Action"][(-1*amount)+i]])
		#oldest in the list is first
	return recom


def sustain(ticker): #gets sustainability
	dictionary = ticker.sustainability.to_dict()
	sus = []
	for i in dictionary:
		for j in dictionary[str(i)]:
			# (the "value" like militaryContract and that value such as false e.i. dictionary["Value"]["militaryContract"] is false
			sus.append([j, str(dictionary[str(i)][str(j)])])
	return sus
def news(ticker):
    news = ticker.news
    other = []
    for i in news:
        if str(i["type"]) != "VIDEO":
            #(title,publisher, time of publishment)
            other.append([i["title"],i["publisher"],i["providerPublishTime"]])
    return other


def nyt_paragraphs(url):
        x = get_page(str(url)).find_all("p")
        arr = []
        for i in range(6,len(x)-1):
                arr.append(x[i])
        return arr

def get_nyt(date, nyt_key): #nyt archive api currently
	#date should be something like /2021/11  /<year>/<month>
	new_york_times = requests.get("https://api.nytimes.com/svc/archive/v1"+date+".json?api-key="+str(nyt_key))
	try:
		new_york_times = new_york_times.json()["response"]["docs"]
	except:
		print('Didnt get a response')
		print('sleeping for a minute')
		time.sleep(60)
		return get_nyt(date, nyt_key)
	return pd.json_normalize(new_york_times)


	"""
	new_york_times is now a array of dictionaries
	to print out everything and see what you need, copy and paste this into another program (of course import the function first)

	for i in get_nyt("/2021/11",<your api key>):
	for j in i.items():
		print(j)
	print("\n\n\n ========================\n\n\n")

	^ prints out every dictionary (key, value) as a list

	"abstract"
	"web_url"
	"snippet"
	"lead_paragraph"

	this can be None
	"headline"
	    "main"
	    "kicker"
	    "content_kicker"
	    "print_headline"
	    "name"
	    "seo"
	    "sub"

	 the subsections of keywords are lists
	"keywords"
	    "name"
	    "value"
	    "rank"
	    "major"

	"pub_date"
	"document_type"
	"news_desk"
	"section_name"
	"byline"
		"original"
		"person"
		        "firstname"
		        "middlename"
		        "qualifier"
		        "title"
		        "role"
		        "organization"
		        "rank"
		"organization"

	"type_of_material"
	"word_count"


	^^ is everything I think would play any use
	"""
if  __name__ == '__main__':
	get_nyt('/2020/9', "")
#448106
