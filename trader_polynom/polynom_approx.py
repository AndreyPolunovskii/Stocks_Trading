import pandas as pd
from scipy.optimize import curve_fit
import numpy
import math as mth
from mymetric import (my_mean_sqeared_error,my_average)


###############################
def function(x,n,kward):
    v = kward[0]
    for i in range(1,n):
        v += kward[i] * (x ** i)
    return v
###############################
def make_func(n):
    def funct(x,*kward):
        v = kward[0]
        for i in range(1,n):
            v += kward[i] * (x ** i)
        return v
    return funct

###############################
# Mass_df = preprocess_mass(tickers,t,init_point)
def polynom_approx(X,Y,col_predict_point = 10,col_verify_point = 0,order_polynom = 5,p0_1 = 0.2,print_atrib = False):

    # делим массивы на " обучающую " , " проверяемую " и "предсказательную" выборки
    m = col_verify_point # количество проверяемых точек
    l = col_predict_point

    X_fit = X[:-(m+l)]
    Y_fit = Y[:-m]

    #порядок полинома
    n = order_polynom

    #задаем параметры начального вектора
    p0 = numpy.zeros(n)
    p0[0] = my_average(Y,numpy.zeros(len(Y))) 
    p0[1] = 0.2
    for i in range(1,n):
        p0[i] = p0[1] ** i

    if print_atrib == True:
        print("-------------------")
        print("Начальный вектор p0")
        print(p0)

    # ищем коэффициенты полинома
    params = curve_fit(make_func(len(p0)),X_fit,Y_fit,p0 = p0)

    if print_atrib == True:
        print("Коэффициенты аппроксимирующего многочлена")
        print(params[0])

    # аппроксимированная кривая
    approx_Y_t = function(X,len(params[0]),params[0])
    
    return approx_Y_t




