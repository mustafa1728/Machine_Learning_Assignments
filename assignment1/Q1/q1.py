import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class GradientDescent():
    def __init__(self, lr):
        self.lr = lr
        self.X = None
        self.Y = None
        self._theta = None
        self._grad = None
        self._loss_history = []
        self._theta_history = []
        self.ax_3d = None
        self.ax_cnt = None

    def load_data(self, x_path, y_path):
        self.X = pd.read_csv(x_path).values
        self.Y = pd.read_csv(y_path).values
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        self.Y = (self.Y - self.Y.mean(axis=0)) / self.Y.std(axis=0)
        self.X = np.append(np.ones((self.X.shape[0], 1)), self.X.reshape(-1, 1), axis = 1)
        return

    def training_step(self, data):
        (X, Y) = data
        # computing gradient
        self._grad = - np.dot(np.transpose(X), (Y -  np.dot(X, self._theta)))
        self._grad = np.mean(self._grad, axis=1).reshape(-1, 1)
        # optimisation step
        self._theta = self._theta - self.lr * self._grad

    def check_sanity(self):
        # checking whether data loaded properly
        try:
            assert self.X is not None
            assert self.Y is not None
            assert self.X.shape[0] == self.Y.shape[0]
        except AssertionError:
            raise RuntimeError("Data not loaded properly. Please run model.load_data() before training!")

    def train(self, max_iter = 1000000):
        self.check_sanity()

        # Initialising weights (X already has intercept term)
        self._theta = np.zeros((self.X.shape[1], 1))

        print("Intial Loss is {}".format(self.loss()))
        # while self._grad is None or not self.stopping(self.loss()):
        for iter in range(max_iter):
            if self._grad is not None and self.stopping():
                break
            self.training_step((self.X, self.Y))
            if iter % 10 == 0:
                print("Iteration {} | Loss {}".format(iter, self.loss()))
            self._loss_history.append(self.loss())
            self._theta_history.append(self._theta)

        print("Total {} iterations. Final parameters: theta {}".format(iter, self._theta.reshape(-1)))
        print("Final loss is {}".format(self.loss()))

    def stopping(self):
        def stopping_criteria1(grad, threshold = 0.0001):
            # return abs(grad[0][0]) <= threshold and abs(grad[1][0]) <= threshold
            return np.linalg.norm(grad) <= threshold

        def stopping_criteria2(loss, threshold = 0.1):
            return loss <= threshold

        return stopping_criteria1(self._grad)
        # return stopping_criteria2(self.loss())

    def loss(self): 
        squares = np.square(self.Y -  np.dot(self.X, self._theta))
        loss = np.mean(squares) / 2
        return loss

    def _loss_for_plot(self, theta0, theta1):
        no_samples = self.Y.shape[0]
        diffs = [self.Y[i] - theta0 - theta1*self.X[i, 1] for i in range(no_samples)]
        squares = [i**2 for i in diffs]
        loss = sum(squares) / (2 * no_samples)
        return loss

    def _plot_data_hyp(self, save_folder = None):
        x = self.X[:, 1]
        y = self.Y
        plt.scatter(x, y)
        arg_s = np.argsort(x)
        new_x = self.X[arg_s, 1]
        Y_hypth = np.dot(self.X[arg_s], self._theta)

        plt.plot(new_x, Y_hypth, color='red')
        plt.show()
        return
    
    def _plot_loss_3d(self, save_folder = None):
        if self.ax_3d is None:
            self.ax_3d = plt.axes(projection='3d')
            theta0 = np.linspace(-1, 1, 30)
            theta1 = np.linspace(0, 1.2, 30)
            theta0, theta1 = np.meshgrid(theta0, theta1)
            J = self._loss_for_plot(theta0, theta1)

            self.ax_3d.plot_surface(theta0, theta1, J, cmap='viridis')
            self.ax_3d.set_title('Loss vs parameters')
            # plt.show()

        theta0 = np.asarray([i[0] for i in self._theta_history])
        theta1 = np.asarray([i[1] for i in self._theta_history])
        self.ax_3d.plot(theta0.reshape(-1), theta1.reshape(-1), self._loss_history, color = "red", linewidth = 10)
        plt.show()
        return
    
    def _plot_loss_contours(self, save_folder = None):
        if self.ax_cnt is None:
            theta0 = np.linspace(-1, 1, 30)
            theta1 = np.linspace(0, 1.2, 30)
            theta0, theta1 = np.meshgrid(theta0, theta1)
            J = self._loss_for_plot(theta0, theta1)

            fig,self.ax_cnt=plt.subplots(1,1)
            cp = self.ax_cnt.contourf(theta0, theta1, J)
            fig.colorbar(cp)
            self.ax_cnt.set_title('Loss vs parameters')

        theta0 = [i[0] for i in self._theta_history]
        theta1 = [i[1] for i in self._theta_history]
        self.ax_cnt.plot(theta0, theta1, color='red')
        plt.show()
        return

def main():
    model = GradientDescent(lr = 0.001)
    model.load_data("linearX.csv", "linearY.csv")
    model.train()
    model._plot_data_hyp()
    model._plot_loss_3d()
    model._plot_loss_contours()

if __name__ == "__main__":
    main()
