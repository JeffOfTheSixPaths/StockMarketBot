import pandas as pd
import numpy as np

def comparison_preproccess(junk) -> list:
    if type(junk) == type(1.0): return np.asarray(junk)
    '''
    the input is supposed to look like "[1,2,3,4]" which is a string
    the function takes this and outputs [1,2,3,4] which is an array of floats

    to do this the function first removes the first and last character, effectively removing the brackets ("[1,2,3,4]" -> "1,2,3,4")
    nan as well as inf appear in the data, we convert it to a value we want it to be (nan = 0, and inf = 99)
    so we replace all values of "nan" and "inf in the string
    it then splits the string by the commas ("1,2,3,4" -> ["1", "2", "3", "4"]  )
    finally, it converts each string element to it's numerical equivalent as a float
    '''
    arr_without_brackets = junk[1:-1]  #removes first and last character
    arr_without_brackets = arr_without_brackets.replace("nan", "0")
    arr_without_brackets = arr_without_brackets.replace('inf', '99')
    arr = arr_without_brackets.split(",") #splits by commas
    for i,e  in enumerate(arr): #converts to numerical equivalent and saves to the array
        try:
            arr[i] = float(e)
        except:
            raise ValueError(f'Could not convert value {e} to float')
    
    return np.asarray(arr)

evaluation_df = pd.read_csv('IDIDIT.csv', sep = '\t')
evaluation_df['average of the next 4'] = evaluation_df['average of the next 4'].map(comparison_preproccess)
evaluation_df['Prices D7'] = evaluation_df['Prices D7'].map(comparison_preproccess)

lcol = 'Prices D7'
a = 'average of the next 4'

lcol = evaluation_df[lcol]
a  = evaluation_df[a]

n = len(evaluation_df['comparison']) - 1
for i in range(n):
    av = a.loc[i] 
    l = lcol.loc[i][0]
    c = av > l
    print(f'the comparison of {av} > {l} is {c}')
    evaluation_df['comparison'].loc[i] = c