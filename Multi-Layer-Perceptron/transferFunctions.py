import numpy as np 

'''
A set of common transfer functions used in Neural Networks.
'''
def logsig(x):
    return 1 / (1 + np.exp(-x))


def tansig(x):
    return ( np.exp(x) - np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )


def purelin(x):
    return x


def hardlim(x):
    if x < 0:
        return 0
    else:
        return 1
    

def hardlims(x):
    if x < 0:
        return -1
    else:
        return 1


def satlin(x):
    if x < 0:
        return 0
    elif x >= 0 and x <= 1:
        return x
    else:
        return 1


def satlins(x):
    if x < -1:
        return -1
    elif x >= -1 and x <= 1:
        return x
    else:
        return 1


def poslin(x):
    if x < 0:
        return 0
    else:
        return x