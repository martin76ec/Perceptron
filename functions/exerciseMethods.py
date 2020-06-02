import os 
import numpy
from random import randrange
from matplotlib.image import imread

def buildFeatureMatrix(path, age): 
    X = numpy.ones((0, 40000))
    images = os.listdir(path)
    for image in images:
        name = image.split('_')
        if name[0] == str(age): 
            imageArray = imread(os.path.join(path, image))
            imgArrayGrayscale = imageArray[:, :, 0]
            img = numpy.reshape(imgArrayGrayscale, (40000, 1), order='F')
            imgTransposed = numpy.transpose(img)
            X = numpy.concatenate((X, imgTransposed))
    return X

def buildResultArray(path, age):
    images = os.listdir(path)
    count = 0
    for image in images:
        name = image.split('_')
        if name[0] == str(age): 
            count += 1

    y = numpy.empty((count, 1))
    count = 0
    
    for image in images:
        name = image.split('_')
        if name[0] == str(age): 
            #y = numpy.append(y, [[int(name[1])]])
            y[count, 0] = int(name[1])
            count += 1
    return numpy.array(y)

def buildRandomWeight(length): 
    w = numpy.zeros((length + 1, 1))
    for i in range(length):
        w[i, 0] = randrange(0, 5)
    return numpy.array(w)