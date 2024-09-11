# coding: utf-8

# linear_regression/test_temperature_polynomial.py

import regression

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import time



def fr(costs, subfile):
        min = 0
        internal = len(costs) 
        srcX, y = regression.loadDataSet(costs);
        srcX = srcX.T

        m,n = srcX.shape
        p = 4
        for i in range(p):
            srcX = np.concatenate((srcX, np.power(srcX[:, 0], i+2)), axis=1)

   
        X = regression.standarize(srcX.copy())
        # X = srcX/20
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
        rate = 0.1
        maxLoop = 1000
        epsilon = 0.01

        result, timeConsumed = regression.bgd(rate, maxLoop, epsilon, X, y)
        theta, errors, thetas = result

 
        fittingFig = plt.figure()
        title = subfile
        ax = fittingFig.add_subplot(111, title=title)
        trainingSet = ax.scatter(srcX[:, 0].flatten().A[0], y[:,0].flatten().A[0], s=1, c='green', marker='v')


        k = 10 
        xx = np.linspace(min, min + internal, k)

        xx2 = np.power(xx, 2)
        xx3 = np.power(xx, 3)
        xx4 = np.power(xx, 4)
        xx5 = np.power(xx, 5)

        yHat = []

        for i in range(k):
            normalizedSize = (xx[i] - xx.mean()) / xx.std(0)
            normalizedSize2 = (xx2[i] - xx2.mean()) / xx2.std(0)
            normalizedSize3 = (xx3[i] - xx3.mean()) / xx3.std(0)
            normalizedSize4 = (xx4[i] - xx4.mean()) / xx4.std(0)
            normalizedSize5 = (xx5[i] - xx5.mean()) / xx5.std(0)

            x = np.matrix([[1, normalizedSize, normalizedSize2, normalizedSize3, normalizedSize4, normalizedSize5]])
            yHat.append(regression.h(theta, x.T))

        fittingLine, = ax.plot(xx, yHat, linewidth='1', color='brown')

        ax.set_xlabel('tasks')
        ax.set_ylabel('cpu_request')

        plt.legend([fittingLine, trainingSet],['five Regression', 'Training Set'])
        plot_file = subfile + '.png'
      
        plt.show()

        # return first_diff


def difference(dataset, interval=1):
        diff = []
        d = interval
        for d in range(interval, len(dataset)):
            value = dataset[d] - dataset[d - interval]
            diff.append(value/interval)
        return diff



