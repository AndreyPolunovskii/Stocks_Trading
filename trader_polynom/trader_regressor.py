import pandas as pd
import os, time
from polynom_approx import polynom_approx
from pre_postprocessing import (pre_post_proc_approx,preprocess_mass)


df = pd.DataFrame(columns=['time','Main Price','Prev Close','Open','Bid','Ask','Volume'])

os.environ['TZ']='America/New_York'
time.tzset()
t =time.localtime()
clear = lambda: os.system('clear')


###################################
#управляющие параметры

               #на скольки точках тестим
               #нулевые значения нельзя тож подставлять!!
param_points={'test_col':1,
               # на сколько точек предсказываем (точек должно быть как минимум 10 желательно)
              'point_pred':3,
               # сколько обученных точек показываем
               'train_point_print':47 }

#порядок инетрполирующего многочлена
order_polynom = 4

#сколько последних моментов времени взять из csv файла для построения регрессии
# не должно быть меньше чем сумма test_col+train_point_print!!!
init_point = 50

# шаг по времени ( если использовать шаг по времени парсера, принять -1 )
freq = -1

#какие компании обрабатываем
tickers= ['IQ','AAPL','GOOG']  #['FB','AAPL','GOOG','TWTR','FB','IQ','AMZN']

#интервал пересчета регрессии
freq_reg = 60*10


#рисовать графики (1) и нерисовать графики (0)
check_drow = 1


#много времени занимает обработка нескольких компаний
####################################

###############
fvar=open('VAR.txt')
lines = fvar.readlines()

if freq == -1:
  freq = int(lines[0])

fvar.close()
###############

# достаем датасет
Mass_df = preprocess_mass('data/full_data/',tickers,init_point)

#while(1):  #t.tm_hour < 16):
clear()
for tick in tickers:
    pre_post_proc_approx(tick,Mass_df[tick],param_points,freq,t,order_polynom)
 # time.sleep(freq_reg)


