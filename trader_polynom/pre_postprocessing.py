import urllib.request , time ,  re
import pandas as pd
import numpy  ,os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import dates
from matplotlib.ticker import FormatStrFormatter ,AutoMinorLocator ,MultipleLocator
import datetime as dt
import math
import requests
from polynom_approx import polynom_approx
from mymetric import *



####################################################
def preprocess_mass(direct,strings,init_point):
    Mass_DF={}

    for str1 in strings:
   #     Mass_DF[str1]=pd.read_csv('data/'+ str(t.tm_mday)+"-"+str(t.tm_mon)+"-"+str(t.tm_year)+'/' + str1 + '.csv')
        Mass_DF[str1] = pd.read_csv(direct+str1+'.csv')
        Mass_DF[str1] = Mass_DF[str1].iloc[-init_point:]
        Mass_DF[str1].index = Mass_DF[str1]['time']
        del Mass_DF[str1]['time'] # убираем созданный в csv файле столбец с датой и временем

        Mass_DF[str1].index = pd.to_datetime( Mass_DF[str1].index )
          
        Mass_DF[str1]["all_sec"] = Mass_DF[str1].index.second
        
        buf_ind = Mass_DF[str1].index[0]
        Mass_DF[str1].loc[[buf_ind],'all_sec'] = buf_ind.hour * 3600 + buf_ind.minute * 60 + buf_ind.second
    
        for ind in Mass_DF[str1].index:
           delta = ind.hour * 3600 + ind.minute * 60 + ind.second - (buf_ind.hour * 3600 + buf_ind.minute * 60 + buf_ind.second)
           if ind.day != buf_ind.day:
             Mass_DF[str1].loc[[ind],'all_sec']  = int(Mass_DF[str1].loc[[buf_ind],'all_sec']) + delta + 6.5 * 3600
           else:
             Mass_DF[str1].loc[[ind],'all_sec']  = int(Mass_DF[str1].loc[[buf_ind],'all_sec']) + delta
           buf_ind = ind
                        
        
    return Mass_DF
#######################################################
MKdir_gr = lambda t: os.system('mkdir -p graphics/'+ str(t.tm_mday)+"-"+str(t.tm_mon)+"-"+str(t.tm_year) )
#######################################################

def pre_post_proc_approx(str1,df,param_points,delta_t,t,order_polynom):

    X = df['all_sec'].values # преобразовали тип dataFrame в тип array Numpy
    Y = df['Main Price'].values
    time_moments =  list(df.index)    
    

    test_col= param_points['test_col']
    point_pred = param_points['point_pred']
    train_point_print = param_points['train_point_print']

    # добавляем предсказательные точки
    X_buf = numpy.zeros(point_pred)
    
    for i in range(point_pred):
        X_buf[i] = X[X.shape[0] - 1] + (i+1) * delta_t
        time_new_moment = pd.Timestamp( time_moments[len(time_moments)-1] + pd.Timedelta(seconds=delta_t) )
        time_moments.append( time_new_moment )

    X = numpy.hstack( (X, X_buf) )
    
    # делаем прогноз
    approx_Y = polynom_approx(X,Y,point_pred,test_col,order_polynom)

    #функция рисования
    drowing_picture(str1,time_moments,X,Y,approx_Y,test_col,point_pred,train_point_print,t,delta_t)

    return 1

############################################################################


def drowing_picture(str1,time_moments,X,Y,approx_Y,test_col,point_pred,train_point_print,t,delta_t,check_drow = 1):
    

    col_p = point_pred + test_col + train_point_print
    col_t = test_col + train_point_print
 
    average_zn =  my_single_average(Y)
    # оцениваем качество предсказаний
    accuracy = my_mean_sqeared_error(Y , approx_Y[:-point_pred] )
    acc_str = "mean squared error: %.4f%%" % (accuracy*100/average_zn)
   # print(acc_str)

    ma = my_average(Y , approx_Y[:-point_pred] )
    ma_str = "average error: %.3f%%" % (ma*100/average_zn) 
 #   print(ma_str)

    mde,tuk = max_delta(Y, approx_Y[:-point_pred] )
    mde_str = "max delta error: %.3f%%" % (mde*100/average_zn) 

    #тут указываем количество отметок времени на оси x
    col_major_tickers = 15

    # рисуем
    if check_drow:
      fig = plt.figure(figsize=(12, 8))
      ax = fig.add_subplot(111)

      text1 = acc_str + '\n' + ma_str +'\n' +mde_str

      ax.text(0.02, 0.1, text1, bbox=dict(facecolor='white', alpha=0.7), transform=ax.transAxes, fontsize=12)


#####################################
# убираем пустые значения ( например выходные )
      def equidate_ax( ax,x, dates, fmt="%Y-%m-%d , %H:%M:%S"): 
        N = len(dates) 
        def format_date(index, pos=None): 
          if int(index) in x:
             return dates[int(index)].strftime(fmt) 
          else:
             return ""

        ax.xaxis.set_major_locator(plt.MaxNLocator(col_major_tickers))
        ax.xaxis.set_major_formatter(FuncFormatter(format_date)) 
        
#####################################
 
      # рисуем предсказываемое поведение кривой 
      x = numpy.arange( col_p )
      plt.xlim(0,col_p + 1)
      equidate_ax( ax,x, time_moments[-col_p:] ) 
      ax.plot(x, approx_Y[-col_p:], 'r-', label="predict", linewidth=2)

      # рисуем реальное поведение кривой 
      x1 = numpy.arange( col_t )
      ax.plot(x1, Y[-col_t:], 'bo-', label="dataset", linewidth=1)
      
      plt.axvline(x=x[-(point_pred+1):-point_pred], color='k', linestyle='--', label='bound_train', linewidth=2)

  #    plt.axvline(x=time_moments[test_col + train_point_print - 1], color='g', linestyle='--', label='bound_test',
      #          linewidth=2)

      def price(x):
        return "$"+"%.5f" % x

      ax.set_ylabel('price stock')
      ax.set_xlabel('time (h:m:s)')

      ax.format_ydata = price

      majorFormatter = FormatStrFormatter('%.1f$')
      ax.yaxis.set_major_formatter(majorFormatter)

      minorLocator = AutoMinorLocator(n=2)
      ax.xaxis.set_minor_locator(minorLocator)

      ax.set_title('стоимость акций ' + str1)

      for label in ax.xaxis.get_ticklabels(minor=True):
        label.set_rotation(30)
        label.set_fontsize(10)

      for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
        label.set_fontsize(10)

      # надо будеть дать возможность самому пользователю легенду двигать !!
      ax.legend(bbox_to_anchor=(0,1), loc='upper left',ncol=1)

      

    # рисуем сетку
      ax.grid(True, which='major', color='grey', linestyle='dashed')
      ax.grid(True, which='minor', color='grey', linestyle='dashed')

      fig.autofmt_xdate()

   #   plt.show()
      MKdir_gr(t)
      fig.savefig('graphics/'+ str(t.tm_mday)+"-"+str(t.tm_mon)+"-"+str(t.tm_year)+'/цена акции компании '+str1+ '.pdf',format = 'pdf',dpi=1000)


      fig.clf()

    #все эти свободные числа нужно вывести как управляющие параметры
 #   last_price = my_single_average(y_pred[-4:]) #y_pred[-3:-2] #

 #   av_Y = my_single_average(Y[-(test_col):])

    #считаем Гауссову вероятность
  #  if abs(ma) > abs(av_Y):
  #     P = Gauss_probability(0.1,abs(ma),accuracy,mde)
 #   else:
 #      P = Gauss_probability(abs(1-abs(ma/av_Y)),abs(ma),accuracy,mde)

    #выводим на экран данные
 #   print(str1 +": procent %.3f%% of price in %d:%d:%d, probability: %.3f%% " % (last_price,time_interval[-3:-2]['hour'],time_interval[-3:-2]['minute'],time_interval[-3:-2]['sec'], P * 100) )
      diff = (approx_Y[approx_Y.shape[0]-1] - Y[Y.shape[0]-1]) * 100/approx_Y[approx_Y.shape[0]-1] 
      if diff > 0:
          print('"' + str(time_moments[len(time_moments) - 1]) + '" стоимость акции ' + str1 +" будет равна : %.2f" % float(approx_Y[approx_Y.shape[0]-1])+"$"+" (увеличится на %.2f%%)" % abs(diff))
      if diff < 0:
          print('"' + str(time_moments[len(time_moments) - 1]) + '" стоимость акции ' + str1 +" будет равна : %.2f" % float(approx_Y[approx_Y.shape[0]-1])+"$"+" (уменьшится на %.2f%%)" % abs(diff))
        
      if diff == 0:
          print('"' + str(time_moments[len(time_moments) - 1]) + '" стоимость акции ' + str1 +" будет равна : %.2f" % float(approx_Y[approx_Y.shape[0]-1])+"$"+" (осталось такой же как сейчас)")
        
          
          
      
##########################################################
