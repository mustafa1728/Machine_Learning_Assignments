from numpy.core.fromnumeric import shape
from Q1.q1 import GradientDescent

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(GradientDescent):
    def __init__(self, lr):
        super().__init__(lr)

    def load_data(self, x_path, y_path):
        self.X = pd.read_csv(x_path).values
        self.Y = pd.read_csv(y_path).values
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        self.X = np.append(np.ones((self.X.shape[0], 1)), self.X, axis = 1)
        # print(self.X.shape)
        # print(self.Y.shape)
        return

    def loss(self):
        sigmoid = 1/(1+np.exp(-np.dot(self.X, self._theta))) 
        term1 = np.dot((-self.Y).T, np.log(sigmoid))
        term2 = np.dot((1-self.Y).T, np.log(1-sigmoid))
        loss = (term1 - term2) / self.Y.shape[0]
        return loss[0][0]

    def training_step(self, data):
        (X, Y) = data
        # computing hessian
        sigmoid = 1/(1+np.exp(-np.dot(X, self._theta))) 
        # hessian = np.mean(sigmoid * (1 - sigmoid) * X * np.transpose(X), axis = 0)
        self.hessian = np.dot(np.transpose(X), X) * np.diag(sigmoid) * np.diag(1 - sigmoid) / X.shape[0]
        hessian_inv = np.linalg.inv(self.hessian)
        # computing gradient
        self._grad = - np.dot(np.transpose(X), (Y -  sigmoid))
        self._grad = np.mean(self._grad, axis=1).reshape(-1, 1)
        # optimisation step
        self._theta = self._theta - self.lr * np.dot(hessian_inv, self._grad)

    def stopping(self):
        # return super().stopping() or np.linalg.norm(self.hessian) < 0.00001
        return np.linalg.norm(self.hessian) < 0.005 or (len(self._loss_history) > 2 and abs(self._loss_history[-1] - self._loss_history[-2]) < 0.0001)
        # return False

    def accuracy(self, X_test=None, Y_test=None):
        if X_test is not None and Y_test is not None:
            X = X_test
            Y = Y_test
        else:
            X = self.X
            Y = self.Y
        sigmoid = 1/(1+np.exp(-np.dot(X, self._theta))) 
        Y = Y.reshape(-1)
        sigmoid = sigmoid.reshape(-1)
        predictions = sigmoid > 0.5
        corrects = [1 for i in range(Y.shape[0]) if Y[i] == predictions[i]]
        return "{:.2f}%".format(len(corrects))

    def plot_decision(self):
        xx = self.X[:, 1].reshape(-1)
        xy = self.X[:, 2].reshape(-1)

        xx, xy = np.mgrid[min(xx) - 0.1:max(xx)+0.1:.01, min(xy)-0.1:max(xy)+0.1:.01]
        grid = np.c_[xx.ravel(), xy.ravel()]
        theta_temp = self._theta[1:, :]

        h = np.dot(grid, theta_temp)
        h = h.reshape(xx.shape)

        print(xx.shape, xy.shape, h.shape)
        boundary_level = 0 - self._theta[0, 0]
        plt.contourf(xx, xy, h, levels=[-10000, boundary_level, 100000], colors = ["red", "green", "blue"], alpha = 0.2)
        plt.contour(xx, xy, h, levels=[boundary_level], colors = ["blue"], linewidths = 2)
        
        lab_1 = np.asarray([i for i in range(self.Y.shape[0]) if self.Y[i] == 1])
        plt.scatter(xx[lab_1], xy[lab_1], c = "g", marker="^")
        plt.scatter(xx[~lab_1], xy[~lab_1], c = "r", marker="x")

        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/decision_boundary.png", dpi = 300)
        plt.close()

    
def main():
    model = LogisticRegression(lr = 0.001)
    model.load_data("logisticX.csv", "logisticY.csv")
    model.train(max_iter = 10000)
    model.plot_decision()
    print(model.accuracy())

main()