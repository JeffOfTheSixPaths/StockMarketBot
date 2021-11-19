## python webscraper
import requests
from bs4 import BeautifulSoup
import json
import yfinance as yf
## lxml parser downloaded
f = open('websites.json')
data = json.load(f)
research =[["headline", "summary", "article"]]
temp = 0



def get_page(url):
	temp = BeautifulSoup(requests.get(str(url)).text,'lxml')
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
		print("Error:spare \"<\" or \">\" character in: \n" + string_with_tags) # if there was a spare, the parsing algorithm would break
	return string_with_tags ##don't want to do string = string_with_tags somewhere just for return string 

def not_exclude(href):
	#finds href already
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
		input("press ENTER to continue... \n") #make shift pause






def get_rec(ticker, amount):
	diction = ticker.recommendations
	recom = []
	for i in range(amount):
		# (firm, to grade, action)
		recom.append([diction["Firm"][(-1*amount)+i],diction["To Grade"][(-1*amount)+i],diction["Action"][(-1*amount)+i]])
		#oldest in the list is first
	return recom
		
		
		

#448106
