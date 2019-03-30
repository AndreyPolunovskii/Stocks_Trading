import urllib.request , time ,  re
import pandas as pd
import numpy , xgboost
from sklearn.model_selection import GridSearchCV  # ругается что эту функцию скоро уберут из пакета
from sklearn.metrics import (mean_squared_error,confusion_matrix)
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.ticker import FormatStrFormatter ,AutoMinorLocator ,MultipleLocator
import datetime as dt


def par_fin(ticker):
    my_file = open("info","w")
    url="https://finance.yahoo.com/quote/"+ticker+"?p="+ticker
    txt=urllib.request.urlopen(url).read()
    inform = str(txt)
    my_file.write(inform)
    my_file.close()

    l={}

    Main_price = re.search('b\)" data-reactid="21">(.*?)<', inform)
    if Main_price:
        tmp = Main_price.group(1)
        tmp = tmp.replace(",", "")
        l.update({"Main Price":float(tmp)})

    Close = re.search('s\) " data-reactid="15">(.*?)<', inform)
    if Close:
        tmp = Close.group(1)
        tmp = tmp.replace(",", "")
        l.update({"Prev Close":float(tmp)})

    Open = re.search('s\) " data-reactid="20">(.*?)<', inform)
    if Open:
        tmp = Open.group(1)
        tmp = tmp.replace(",", "")
        l.update({"Open":float(tmp)})

    Bid = re.search('s\) " data-reactid="25">(.*?)<',inform)
    if Bid:
        tmp=Bid.group(1)
        tmp = tmp.replace(",", "")
        l.update({"Bid":tmp})


    Ask = re.search('s\) " data-reactid="30">(.*?)<', inform)
    if Ask:
        tmp = Ask.group(1)
        tmp = tmp.replace(",", "")
        l.update({"Ask":tmp})


    Volume = re.search('s\) " data-reactid="43">(.*?)<', inform)
    if Volume:
        tmp = Volume.group(1)
        tmp = tmp.replace(",","")
        l.update({"Volume":int(tmp)})
    else:
        l.update({"err":'error'})
        return l

    return l

####################################################
def add_time(ticker):
    mass=par_fin(ticker)
    t=time.localtime()
    mass.update({"time":str(t.tm_mday)+"-"+str(t.tm_mon)+"-"+str(t.tm_year)+"  "+str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)})
    return mass
####################################################
def preprocess_mass(strings,d_t,df):
    Mass_DF={}
    for i in range(len(strings)):
        Mass_DF.update({strings[i]: pd.DataFrame(df)})

    for str in strings:
        Mass_DF[str]=pd.read_csv('/usr/data/trade1/old_data/' + str + '.csv')

        Mass_DF[str].index = Mass_DF[str][d_t]
        del Mass_DF[str][d_t]  # убираем созданный в csv файле столбец с датой и временем

        Mass_DF[str].index =Mass_DF[str].index.to_datetime()

        Mass_DF[str]["hour"] = Mass_DF[str].index.hour
        Mass_DF[str]["minute"] = Mass_DF[str].index.minute
        Mass_DF[str]["sec"] = Mass_DF[str].index.second


        for n in range(Mass_DF[str].shape[0]):
            lask = Mass_DF[str]['Ask'][n].split()
            lbid = Mass_DF[str]['Bid'][n].split()

            lask[0] = lask[0].replace(",","")
            lbid[0] = lbid[0].replace(",", "")
            Mass_DF[str]['Ask'][n] = float(lask[0])
            Mass_DF[str]['Bid'][n] = float(lbid[0])


    return Mass_DF
#######################################################
def solving(Mass_df,str,delta_t):
  #  del Mass_df[str]['Unnamed: 0']
    Mass_df[str]['diff_price'] = Mass_df[str]['Bid'].diff(1) # изменение цены
  #  print(Mass_df[str][['diff_price','Main Price']])


    print(Mass_df[str].columns)



    dataset = Mass_df[str].values #преобразовали тип dataFrame в тип array Numpy

    X = dataset[1:,5:8]

    #рассматриваем увеличение или уминьшение цены!!
    Y = dataset[1:,8]

    #на скольки точках тестим
    test_col=8

    # на сколько точек предсказываем
    point_pred = 6

    # сколько обученных точек показываем
    train_point_print = 7

    col_p = point_pred + test_col + train_point_print

    #разделяем X и Y на обучающий и тестовый набор данных
    X_train = X[:-test_col]
    y_train = Y[:-test_col]

    model = xgboost.XGBRegressor(max_depth=6,
                                 learning_rate=0.9,
                                 n_estimators=120,subsample=0.9)
                              #  ,
                              #   c,
                             #    )

    # обучаем модель
    model.fit(X_train,y_train)#,eval_metric='rmse')

    print(model) #параметры модели

    delta_t = 20

    # dobavluem predskaz (pervoe slagaemoe 'to colichestvo cdelok ,vtoroe -vrema v sekundax)
    for i in range(point_pred):
        sum_sek = X[X.shape[0] - 1][2] + 60 * X[X.shape[0] - 1][1] + 3600*X[X.shape[0] - 1][0] + delta_t
        X = numpy.vstack([X, [  sum_sek//3600,(sum_sek%3600)//60, (sum_sek%3600)%60  ] ])


    # делаем прогноз
    y_pred = model.predict(X[-col_p:])
  #  print(y_train)
 #   print(model.predict(X_train))
 #   print(X_pred)
 #   print(y_pred)


    DF_print =pd.DataFrame({"price_pred":list(y_pred[-point_pred:]),"time_sec":list(X[-point_pred:,1])})

    y_pred1 = y_pred.copy()
    for i in range(point_pred):
        y_pred1 = numpy.delete(y_pred1, y_pred1.shape[0] - 1)

    #делаем вид времени

    time_interval= pd.DataFrame({"hour":X[-col_p:,0],"minute":X[-col_p:,1],"sec":X[-col_p:,2]})
    time_interval['date']= time_interval['hour'].astype('str')+":"+time_interval['minute'].astype('str')+":"+time_interval['sec'].astype('str')
    fmt = dates.DateFormatter("%H:%M:%S")
    time_interval1 = [dt.datetime.strptime(i,"%H:%M:%S") for i in time_interval['date']]

    #оцениваем качество предсказаний
    accuracy = mean_squared_error(Y[-(test_col + train_point_print):],y_pred1)
    print("squared_error: %.2f%%" % (accuracy * 100))


    #рисуем
    fig ,ax= plt.subplots(figsize=(10,6))
    ax.plot(time_interval1,y_pred,'r-',label="predict",linewidth=2)
    ax.plot(time_interval1[:-point_pred], Y[-(test_col + train_point_print):], 'bo--', label="test", linewidth=1)
  #  ax.plot(time_interval1[:-point_pred], Y[-(test_col + train_point_print):], 'bo', label="test", linewidth=2)

    plt.axvline(x=time_interval1[train_point_print  - 1], color='k', linestyle='--', label='bound_train')
    plt.axvline(x=time_interval1[ test_col + train_point_print - 1],color = 'k',linestyle='--',label='bound_test')

    def price(x):
        return "$%.5f" % x

    ax.format_ydata = price
    ax.xaxis.set_major_formatter(fmt)

    majorFormatter = FormatStrFormatter('%.3f $')
    ax.yaxis.set_major_formatter(majorFormatter)

    minorLocator = AutoMinorLocator(n=2)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_minor_formatter(fmt)

    for label in ax.xaxis.get_ticklabels(minor=True):
        label.set_rotation(30)
        label.set_fontsize(10)

    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
        label.set_fontsize(10)


    ax.legend(loc='upper right')
    #рисуем сетку
    ax.grid(True, which='major', color='grey', linestyle='dashed')
    ax.grid(True,which='minor',color='grey',linestyle='dashed')

   # fig.autofmt_xdate()
    plt.show()
#    print(confusion_matrix(y_test, y_pred))

    return 0