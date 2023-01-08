## python webscraper
import requests
from bs4 import BeautifulSoup as bs
import json
import yfinance as yf
import pandas as pd
import time
## lxml parser downloaded
def get_page(url: str):
	return bs(requests.get(url), 'lxml')

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
		print('nyt didnt get a response retrying in a minute\n')
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


	^^ is everything I think would be any use
	"""




def get_guardian(from_date = None, phrases = ["economy"], api_key = "") -> list:
	url = "https://content.guardianapis.com/search?q=" #this is the base url onto which arguments must be passed
	#phrases are the terms that must be found within the search results in order to be gotten
	if(from_date == None): print("There is no from date")
	#from date needs to be like 2014-01-01

	#this function is needed to say "if you couldn't get the data, try again in 60 seconds"
	def get_url(url:str) -> dict:
		response = requests.get(url)
		try:
			return response.json()['response']['results']
		except:
			print("timeout: trying again")
			time.sleep(60)
			get_url(url)
		

	
	'''
	the phrases list can be manipulated in order to be easily integrated into the url

	to do this we'll turn the list into the string, remove the brackets, turn all spaces into %20 (the space symbol used in urls),
	and turn all commas into %20OR%20 (the %20 on either side are required by the guardian api)
	'''
	
	phrase_search = str(phrases) #turning into a string
	phrase_search = phrase_search[1:-1] #removes the first and last character of the string which are the brackets
	phrase_search = phrase_search.replace(", ", ",") #removes spaces after commas because without it could lead to a broken query
	phrase_search = phrase_search.replace(" ,", ",")
	phrase_search = phrase_search.replace(" ", "%20")
	phrase_search = phrase_search.replace(",", "%20OR%20") 

	url += phrase_search #adds the phrase arguments to the end of the url

	if(not from_date == None): url += f'&from-date={from_date}' #if there's a date to add, add the date argument
	url += "&page-size=200" #makes returns 200 requests maximum per request. You cannot request more than 200 per request
	url += f"&api-key={api_key}" #add the api key argument


	print('getting page 1')
	#there are multiple pages, so we need to make multiple requests
	#theguardian_results stores the results while theguardian stores the entire first response which includes more data such as the amount of pages
	theguardian_results = []
	theguardian = requests.get(url).json()['response'] #this variable is needed in order to see how many pages there are
	theguardian_results.append(theguardian['results'])

	for page_num in range(1, int(theguardian['pages']) + 1): #the +1 is because range() is not inclusive of the endpoint
		tmp_url = url
		tmp_url += f'&page={page_num}' #makes it query more pages

		print(f'getting page {page_num} of the guardian \n')
		results = get_url(url)
		theguardian_results.append(results)

	print('got all the results for the guardian\n')
	#theguardian_results is a 2d array, but needs to be 1 dimensional
	# to do this, we'll flatten the array

	flattened_guardian = []
	for i in theguardian_results:
		for j in i:
			#the time is writen as YY-MM-DDTHH:MM:SS, I need to convert them to just YY-MM-DD
			k = j
			time_modified = k['webPublicationDate'].split("T")[0]
			k['webPublicationDate'] = time_modified
			flattened_guardian.append(k)

	return flattened_guardian

if  __name__ == '__main__':
	phrases = ["economy", "stock", "investment", "inflation","unemployment","recession","depression","economic", "short sell", "shorting", "billion", "million", "shortage", "silicon", "oil"
            "surplus", "merge", "industry", "company"]

	thing = get_guardian(from_date="2021-11-01", phrases=phrases, api_key= "908093b2-6f42-4774-b2e4-b3c6ec953cbd")
	f = open("guardian_test.json",'w')
	json.dump(thing,f, indent=2)
	f.close()
	#print(get_guardian(phrases=phrases))

#448106
