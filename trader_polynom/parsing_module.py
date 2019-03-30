import urllib.request , time ,  re
import pandas as pd
import numpy , xgboost ,os
from sklearn.model_selection import (StratifiedKFold,cross_val_score,cross_val_predict)  
from sklearn.metrics import (mean_squared_error,confusion_matrix)
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.ticker import FormatStrFormatter ,AutoMinorLocator ,MultipleLocator
import datetime as dt
import math
import requests
from mymetric import *
from bs4 import BeautifulSoup


############################################
def par_fin(ticker,delta_t,dict_proxies):
 #   url="https://finance.yahoo.com/quote/"+ticker+"?p="+ticker

# бесплатная версия позволяет парсить не более 5 раз в минуту и 500 раз в день
    url="https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol="+ticker+"&interval="+delta_t+"&apikey=E09C2BZZIPKNNWQA"
    
    #настраиваем прокси и заголовки     
    headers = {'User-agent':'Mozilla/5.0'}
    proxies = {'http':'http://'+dict_proxies[ticker]}
    
    try:
        response = requests.get(url,headers=headers,proxies=proxies)
    except:
        print('Сработало какое-то исключение, возможно сервер не отвечает')
        time.sleep(30)
        response = requests.get(url,headers=headers,proxies=proxies)

    l={}

    json_file = response.json()
    response.close()

    try:
      current_time = json_file["Meta Data"]["3. Last Refreshed"]
    except:
      print("Слишком часто идет запрос")
      time.sleep(60)
      response = requests.get(url,headers=headers,proxies=proxies)
      json_file = response.json()
      response.close()
      current_time = json_file["Meta Data"]["3. Last Refreshed"]


    open_price = json_file["Time Series ("+delta_t+")"][current_time]["1. open"]
    high_price = json_file["Time Series ("+delta_t+")"][current_time]["2. high"]
    low_price = json_file["Time Series ("+delta_t+")"][current_time]["3. low"]
    close_price = json_file["Time Series ("+delta_t+")"][current_time]["4. close"]
 
    main_price = ( float(open_price)+float(high_price)+float(low_price)+float(close_price) )/4

    # на самом деле тут цена открытия за текущий интервал!!!
    l.update({"Main Price":float(main_price),"time":current_time})

  # l.update({"Main Price":float(tmp)})

    return l

####################################################
def add_time(mass):
    os.environ['TZ'] = 'America/New_York'
    time.tzset()
    t=time.localtime()
    mass.update({"time":str(t.tm_mday)+"-"+str(t.tm_mon)+"-"+str(t.tm_year)+"  "+str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)})
    return mass
####################################################

