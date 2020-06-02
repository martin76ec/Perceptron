import numpy

def perceptron(X, w):
    Xbias = numpy.ones((X.shape[0], 1))
    Xaum = numpy.concatenate((Xbias, X), axis = 1) 
    S = Xaum.dot(w)
    S[S <= 0] = -1
    S[S != -1] = 1
    return S