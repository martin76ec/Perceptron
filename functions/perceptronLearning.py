import numpy
import functions.computeMCE as mce
import functions.perceptronOutput as perceptron

def learn(X, y, w):
    Xbias = numpy.ones((X.shape[0], 1))
    Xaum = numpy.concatenate((Xbias, X), axis = 1) 
    yhat = perceptron.perceptron(X, w)
    error = mce.mce(y, yhat)
    while error != 0:
        for i in range(len(y)):
            if y[i, 0] != yhat[i, 0]:
                w = w + (y[i] * numpy.array([Xaum[i, :]]).T)
        yhat = perceptron.perceptron(X, w)
        error = mce.mce(y, yhat)
    return w