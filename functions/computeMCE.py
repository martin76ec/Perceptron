import numpy

def mce(y, yhat):
    missed = 0
    for i in range(len(y)):
        if y[i, 0] != yhat[i, 0]:
            missed += 1
    error = missed/len(y)
    return error
