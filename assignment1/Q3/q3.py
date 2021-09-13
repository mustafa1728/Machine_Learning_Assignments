import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    '''
        Class Implementing Logistic Regression
        Q3 of COL774 Assignment1

        Part a) Implementation
            The class is intialised without any hyperparameters
            Then data is loaded using model.load_data(x_path, y_path)
            Then training starts with model.train()
            The main parameter update step is in training_step(), which includes the hessian and gradient inverse
            Stopping criteria is in stopping()
            train() drives the training by calling training_step() in each iteration, and saves the parameter and loss history
        
        Part b) Data and Decision Boundary
            Implemented in plot_decision()
            Plots the normalised data
            Also plots the decsion boundary
            Also plots the region where positive and negative are predicted with green and red.
    '''
    def __init__(self):
        self.X = None
        self.Y = None
        self._theta = None
        self._grad = None
        self._loss_history = []
        self._theta_history = []

    def summary(self):
        print("Parameters learnt by Logistic Regression: {}".format(self._theta.reshape(-1)))

    def load_data(self, x_path, y_path):
        # Loading data from files and normalising input features
        self.X = pd.read_csv(x_path).values
        self.Y = pd.read_csv(y_path).values
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        self.X = np.append(np.ones((self.X.shape[0], 1)), self.X, axis = 1)

    def loss(self):
        # Calculating the log-likelihood loss for logistic regression
        sigmoid = 1/(1+np.exp(-np.dot(self.X, self._theta))) 
        term1 = np.dot((-self.Y).T, np.log(sigmoid))
        term2 = np.dot((1-self.Y).T, np.log(1-sigmoid))
        loss = (term1 - term2) / self.Y.shape[0]
        return loss[0][0]

    def training_step(self, data):
        (X, Y) = data
        m = X.shape[0]
        # computing hessian
        sigmoid = 1/(1+np.exp(-np.dot(X, self._theta))) 
        D = np.diag((sigmoid * (1 - sigmoid)).reshape(-1))
        self.hessian = np.dot(np.dot(np.transpose(X), D), X)
        hessian_inv = np.linalg.pinv(self.hessian)
        # computing gradient
        self._grad = - np.dot(np.transpose(X), (Y -  sigmoid))
        self._grad = np.mean(self._grad, axis=1).reshape(-1, 1)
        # optimisation step
        self._theta = self._theta - np.dot(hessian_inv, self._grad)

    def stopping(self):
        return len(self._loss_history) > 2 and abs(self._loss_history[-1] - self._loss_history[-2]) < 10**(-6)*abs(self._loss_history[-2])

    def train(self, max_iter = 1000000):
        # Initialising weights (X already has intercept term)
        self._theta = np.zeros((self.X.shape[1], 1))
        self._loss_history.append(self.loss())
        self._theta_history.append(self._theta)

        for iter in range(max_iter):
            if self._grad is not None and self.stopping():
                break
            self.training_step((self.X, self.Y))
            self._loss_history.append(self.loss())
            self._theta_history.append(self._theta)

    def plot_decision(self, save_dir="plots"):
        # For plotting the decision boundary
        # xx are the x coordinates of input features X
        # xy are the y coordinates of input features X
        xx = self.X[:, 1].reshape(-1)
        xy = self.X[:, 2].reshape(-1)

        xx, xy = np.mgrid[min(xx) - 0.1:max(xx)+0.1:.01, min(xy)-0.1:max(xy)+0.1:.01]
        grid = np.c_[xx.ravel(), xy.ravel()]
        theta_temp = self._theta[1:, :]

        h = np.dot(grid, theta_temp)
        h = h.reshape(xx.shape)

        boundary_level = 0 - self._theta[0, 0]
        # The countourf function is used to draw a filled contour, which gives the background color red vs green
        plt.contourf(xx, xy, h, levels=[-10000, boundary_level, 100000], colors = ["red", "green", "blue"], alpha = 0.2)
        # The contour plot is used to draw the decision boundary. A single level draws a single line.
        contour = plt.contour(xx, xy, h, levels=[boundary_level], colors = ["blue"], linewidths = 2, linestyles='solid')

        lab_1 = np.asarray([i for i in range(self.Y.shape[0]) if self.Y[i] == 1])
        xx = self.X[:, 1].reshape(-1)
        xy = self.X[:, 2].reshape(-1)
        plt.scatter(xx[lab_1], xy[lab_1], c = "g", marker="^", label = "Label 1")
        plt.scatter(xx[~lab_1], xy[~lab_1], c = "r", marker="x", label = "Label 0")

        plt.xlabel("Normalised $x_1$")
        plt.ylabel("Normalised $x_2$")
        plt.title("Logistic Regression Decision Boundary")
        contour.collections[0].set_label("Decision Boundary")
        plt.legend()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "decision_boundary.pdf"), dpi = 300)
        plt.close()

def get_args():
    # for getting command line arguments
    parser = argparse.ArgumentParser(description='Logistic Regression')
    parser.add_argument('--data_path', type=str, default=".", help='the directory path where data files are stored')
    parser.add_argument('--save_path', type=str, default="plots", help='the directory path where plots will be saved')
    args = parser.parse_args()
    return args
    
def main():
    args = get_args()

    model = LogisticRegression()
    model.load_data(os.path.join(args.data_path, "logisticX.csv"), os.path.join(args.data_path, "logisticY.csv"))
    model.train(max_iter = 10000)
    model.summary()
    model.plot_decision(args.save_path)

main()