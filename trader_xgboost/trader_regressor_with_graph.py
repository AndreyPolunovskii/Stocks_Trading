import pandas as pd
import os
from my_finance_module import (time,add_time,boosting_solver, preprocess_mass,fourier_transform)

import warnings
pd.options.mode.chained_assignment = None # отключаем ненужные исключения
warnings.simplefilter(action='ignore',category = FutureWarning) 
warnings.simplefilter(action='ignore',category = RuntimeWarning)
df = pd.DataFrame(columns=['time','Main Price','Prev Close','Open','Bid','Ask','Volume'])

os.environ['TZ']='America/New_York'
time.tzset()
t =time.localtime()
clear = lambda: os.system('clear')


###################################
#управляющие параметры

               #на скольки точках тестим
param_points={'test_col':5,
               # на сколько точек предсказываем (точек должно быть как минимум 10 желательно)
              'point_pred':11,
               # сколько обученных точек показываем
               'train_point_print':12 }

#параметры модели Бустинга
param_model = { 'max_depth':4,
                'learning_rate':0.5,
                'n_estimators':80,
                'subsample':0.05 }


# шаг по времени ( если использовать шаг по времени парсера, принять -1 )
freq = -1

#какие компании обрабатываем
tickers= ['IQ','AAPL','GOOG']  #['FB','AAPL','GOOG','TWTR','FB','IQ','AMZN']

#интервал пересчета регрессии
freq_reg = 60*10

#по сколько точек сглаживать входные данные и на их основе предсказывать
n = 1

#по сколько точек сглаживать промоделированные данные
m = 5

#сколько последних моментов времени взять из csv файла для построения регрессии
init_point = 100

#рисовать графики (1) и нерисовать графики (0)
check_drow = 1

#на сколько фреймов разбить данные для кросс валидации ( при значении 0 кросс валидация отключена )
cross_working = 0

#много времени занимает обработка нескольких компаний
####################################

###############
fvar=open('VAR.txt')
lines = fvar.readlines()

if freq == -1:
  freq = int(lines[0])

fvar.close()
###############

while(1):  #t.tm_hour < 16):
  clear()
  for tick in tickers:
    finished_mass = boosting_solver( preprocess_mass(tickers,t,init_point),tick,freq,t,param_points,param_model,n,m,check_drow,cross_working)
   #fourier_transform(preprocess_mass(tickers, 'time', df), company, freq)
  time.sleep(freq_reg)


