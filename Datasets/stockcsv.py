#python
import csv
from datetime import date
from dbm import dumb
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
t0dataX = []
t0dataY = []
m_data = []
def get_csv(ticket, time_frame):
	#time_frame has to be from the program directly, so should be inputed as a string
	ticket=str(ticket)
	yf.Ticker(ticket).history(period = time_frame).to_csv(ticket)  
	
def time_to_num(TempString):
    ##rightmost point is zero
	# 2022 - 3 - 20     splits into ['2022','3','20'] which is year,month,day
    Temp_Year = int(TempString.split('-')[0]) # gets the year portion 
    Temp_Month = int(TempString.split('-')[1]) # gets the month portion
    Temp_Day = int(TempString.split('-')[2]) # gets the day portion
    d0 = date(Temp_Year, Temp_Month, Temp_Day)
    d1 = date(year0,month0,day0)
    delta = d1 - d0
    return delta.days

def into_graph(ticker): #plot no plot means do you want to be shown the graph
	get_csv(ticker,"6mo")
	df = pd.read_csv(ticker) #was CSV_list
	#data = list(csv.reader(open(ticker,'r')))
	global year0,month0,day0
	## syntax: print(data[n][k]) n>0 k>=0 n is the row and k is the portion or colum within that row
	nop  = df["Date"][0]
	year0 = int(nop.split('-')[0])
	month0 = int(nop.split('-')[1])
	day0 = int(nop.split('-')[2])
	tdata_size = df.shape[0]

	for i in range((int(tdata_size) - 1)):	
		t0dataX.append(abs(time_to_num(df["Date"][i+1])))
		t0dataY.append(float(df["Open"][i+1]))

		    
		x = np.array(t0dataX)
		
		y = np.array(t0dataY)

		m,b = np.polyfit(x,y,1)
		m_data.append(m)
	
	return list(zip(t0dataX,t0dataY))

def nn_csv(ticker,n=7,m=3):
	ticker = str(ticker)
	n = int(n)
	m = int(m)
	#name is '<ticker> n = <n> m = <m>'
	name = ticker + " n= " + str(n) + " m= "+ str(m)
	if name in os.listdir():
		raise Exception("that csv already exists")
	f = open(name + ".csv" , 'a')
	#creates the header at the top
	for i in range(n):
		f.write("value "+str(i)+",")
	f.write("avg of next"+str(m)+',')
	f.write('comparison\n')
	
	#now to get the data into the csv
	data = into_graph(ticker)
	max = len(data) - m -1
	values = []
	future_values = []
	for i in range(n):
		values.append(data[i][1]) # should append the first n values into the values list
	for i in range(n,n+m):
		future_values.append(data[i][1])
	
	#writes the first values into the csv
	for value in values:
		f.write(str(value) + ",")
	sum = 0
	for value in future_values:
		sum+=value/(len(future_values))
	f.write(str(sum)+",")
	if sum > values[-1]:
		f.write("down\n")
	else:
		f.write("up\n")

	#writes everything past the first line of data
	for i in range(n,max):
		values.pop(0)
		future_values.pop(0)
		values.append(data[i][1])
		future_values.append(data[i+m][1])


		for value in values:
			f.write(str(value) + ",")
		sum = 0
		for value in future_values:
			sum+=value/(len(future_values))
		f.write(str(sum)+",")
		if sum > values[-1]:
			f.write("down\n")
		else:
			f.write("up\n")
	f.close()
	f.close()
	print("done printing " + str(name))

	
	

		



if __name__ == '__main__':
	nn_csv(sys.argv[1])
