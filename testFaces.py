import numpy
import functions.exerciseMethods as builder
import functions.perceptronOutput as perceptron
import functions.perceptronLearning as perceptronLearning
import functions.computeMCE as mce
from sklearn.metrics import confusion_matrix

X = builder.buildFeatureMatrix('UTKFace', 36)
y = builder.buildResultArray('UTKFace', 36)
train = X[0:50]
test = X[51:71]

y_train = y[0:50]
y_test = y[51:71]
y_train[ y_train == 0 ] = -1
y_test[ y_test == 0 ] = -1

#Initial Error
w = builder.buildRandomWeight(train.shape[1])
yhat_train = perceptron.perceptron(train, w)
error_initial = mce.mce(y_train, yhat_train)
print('#### TRAINING ####\n')
print('Initial error: ')
print(error_initial)

#Train error
best_w = perceptronLearning.learn(train, y_train, w)
yhat_train = perceptron.perceptron(train, best_w)
error_train = mce.mce(y_train, yhat_train)
print('Train error: ')
print(error_train)
print('Train CM: ')
print(confusion_matrix(y_train, yhat_train))

#Test error
yhat_test = perceptron.perceptron(test, best_w) 
error_test = mce.mce(y_test, yhat_test)
print('\n\n#### TESTING ####\n')
print('Test_error')
print(error_test)
print('Test CM: ')
print(confusion_matrix(y_test, yhat_test))




