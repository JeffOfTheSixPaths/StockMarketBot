import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from datetime import date

def time_to_num(TempString):
    ##rightmost point is zero
    Temp_Year = int(TempString.split('-')[0])
    Temp_Month = int(TempString.split('-')[1])
    Temp_Day = int(TempString.split('-')[2])
    d0 = date(Temp_Year, Temp_Month, Temp_Day)
    d1 = date(year0,month0,day0)
    delta = d1 - d0
    return delta.days
def get_csv(ticket, time_frame):
	#time_frame has to be from the program directly, so should be inputed as a string
	ticket=str(ticket)
	yfinance.Ticker(ticket).history(period = time_frame).to_csv(ticket)    

##------------------------------------------------------
## Code Below
ticker = input("input ticketer: ")
get_csv(ticker, "5d")



CSV_list = 0 #took code from an old project, cant be bothered to change anything right now 
for x in range(len(CSV_list)):
    df = pd.read_csv(CSV_list[x])
    data = list(csv.reader(open(CSV_list[x],'r')))
    ## syntax print(data[n][k]) n>0 k>=0 n is the row and k is the portion or colum within that row
    nop  = data[1][0]
    year0 = int(nop.split('-')[0])
    month0 = int(nop.split('-')[1])
    day0 = int(nop.split('-')[2])
    tdata_size = df.shape[0]
    for i in range((int(tdata_size) - 1)):
        t0dataX.append(abs(time_to_num(data[1+i][0])))
        t0dataY.append(float(data[1+i][1]))

            
    x = np.array(t0dataX)
        
    y = np.array(t0dataY)
        
    plt.plot(x,y)

    m,b = np.polyfit(x,y,1)

    plt.plot(x,m*x+b)
    print(m)
    print(CSV_list)
    plt.show()
    t0dataY.clear()
    t0dataX.clear()
	
print(CSV_list)
print("done")
