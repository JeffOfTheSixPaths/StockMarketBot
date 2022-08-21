## python webscraper
import requests
from bs4 import BeautifulSoup
import json
import yfinance as yf
import pandas as pd
## lxml parser downloaded
research =[["headline", "summary", "article"]]
temp = 0



def get_page(url):
	return BeautifulSoup(requests.get(str(url)).text,'lxml')
	
def get_headlines(site):
	site = str(site)
	if(site == "https://finance.yahoo.com/news/"):
		headlines = get_page("https://finance.yahoo.com/news").find_all("h3")
		return headlines
		
def find_href(string):
	return str(string).split("href=")[1].split("\"")[1]
	
	
	
def get_summaries(site):
	return get_page(str(site)).find_all("p")

def print_summaries(site):
	article = get_summaries(str(site))
	print("====================================")
	for i in range(len(article) - 1):
		if rmAngles(article[i]) != "Related Quotes":
			print(rmAngles(article[i]))
			print("\n\n")
	print("====================================")

def get_articles(site):
	string = rmAngles(str(get_page(str(site)).find_all("div", class_="caas-body")))
	print(string)	
	string = string[:0] + string[1:]
	return string[:-1]
	

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

def not_exclude(href):
	#finds href already
	return href.find("/video/") == -1
	#just for testing under here
	if href.find("/video/") == -1:
			#link = link.split("/")[2]
			return True









def nue_net(link):
	link = str(link)
	headline_list = get_headlines(link)
	summary_list = get_summaries(link)
	print(len(headline_list))
	print(len(summary_list))
	for i in range(2,len(summary_list)):
		temp_list = [rmAngles(headline_list[i+4]),rmAngles(summary_list[i])]
		article_href = find_href(headline_list[i+2])
		article_link = "https://finance.yahoo.com" + article_href
		print(article_link)
		if not_exclude(article_href):
			temp_list.append(get_articles(article_link))
			
		else:
			temp_list.append("<video>")
		research.append(temp_list)
	return research
	



def print_research():
	for i in range(len(research) - 1):
		for index in range(3):
			print(research[i][index])
			print("\n")
		print(len(research) - i - 2)
		input("press ENTER to continue... \n") #just for pausing






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
	new_york_times = new_york_times.json()["response"]["docs"]
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
	nyt_dict = get_nyt("/2021/9","plcRzjMm4wKxhYskXNKOuGufpGpZLK4h")
	print(nyt_dict['pub_date'])
#448106
