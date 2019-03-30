import math

def max_delta(X,Y):
    max_del = 0
    max_elem = 0

    Z= X - Y
    i = 0

    for n in Z:
        if abs(n) > max_del:
            max_del = abs(n)
            max_elem = X[i]
        i += 1

    return abs(max_del),abs(max_elem)
#######################################################
def my_mean_sqeared_error(X,Y):
    mean = 0
    Z= X - Y
    i = 0

    for n in Z:
            mean += pow(n,2)  
    i += 1       
    return math.sqrt(mean/Z.shape[0])
#######################################################
def my_average(X,Y):
    mean = 0
    Z= X - Y
    i = 0
    for n in Z:
            mean += abs(n)
    i += 1
    return abs(mean/Z.shape[0])
#######################################################
def my_single_mean_sqeared(X):
    mean = 0
    for n in X:
            mean += pow(n,2)

    return math.sqrt(mean/X.shape[0])
#######################################################
def my_single_average(X):
    mean = 0
    for n in X:
            mean += n
    return mean/X.shape[0]
#######################################################
def Gauss_probability(coeff,av,av_sqr,x):
    k = - ((x-av)**2/(10*(av_sqr**2)))
    p = math.exp(k) *coeff# /(av_sqr * math.sqrt(2*math.pi))
    return p
#######################################################
