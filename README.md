# StockMarketBot
 A bot that will eventually try to predict stocks.
 saying it's a work in progress is putting it lightly

to do:
  - have the webscraper get more data
  - make the sentiment model better

dependencies:
```
pip install beautifulsoup4
pip install lxml
pip install yfinance
pip install numpy
pip install pandas
pip install tensorflow==2.8.0
pip install tensorflow_datasets
pip install keras
pip install matplotlib
pip install gdown
```

# usage:
  within the Web_Scraping folder is the code that is used to generate the dataset. config.json has different parameters which can be used for the dataset, however only the default parameters have been used for machine learning. running `dataToCsv.py` will generate a csv file `IDIDIT.csv` which uses tabs for seperators. This is the dataset that is used for machine learning. This file takes an extremely long amount of time to run, about 2 hours with an RTX gpu and 32 gigabytes of ram, so goodluck anyone trying to actually run this. 

# The machine learning:
  The model that is being submitted is stored within `ML_Prediction/Multivariate_Time_Series_LSTM/` and the weights are in the `saved model` folder.
