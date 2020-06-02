import numpy
import functions.exerciseMethods as builder
import functions.perceptronOutput as perceptron
import functions.perceptronLearning as perceptronLearning
import functions.computeMCE as mce

U = [[1,0,1],[1,0,1],[1,1,1]]
L = [[1,0,0], [1,0,0], [1,1,1]]
T = [[1,1,1], [0,1,0], [0,1,0]];
Uv = numpy.reshape(U, (9, 1), order='F').T
Lv = numpy.reshape(L, (9, 1), order='F').T
Tv = numpy.reshape(T, (9, 1), order='F').T

X = numpy.concatenate((Uv, Lv, Uv, Tv), axis=0)
w = numpy.array([[1],[0],[1],[1],[1],[1],[1],[1],[1],[0]])

yhat = perceptron.perceptron(X, w)
y = numpy.array([[1],[-1],[1],[-1]])

wfinal = perceptronLearning.learn(X, y, w)

print(wfinal)
