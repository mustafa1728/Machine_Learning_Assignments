import numpy as np
import matplotlib.pyplot as plt
import os


class GDA():
    def __init__(self, boundary):
        self.boundary = boundary

    def load_data(self, x_path, y_path):
        self.X = np.genfromtxt(x_path, dtype= (int,int))
        self.Y = np.genfromtxt(y_path, dtype = "str")
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        # self.X = np.append(np.ones((self.X.shape[0], 1)), self.X, axis = 1)
        self.Y = np.asarray([0 if y == "Alaska" else 1 for y in self.Y])

    def train(self):
        indicator_0 = 1 * (self.Y == 0).reshape(-1, 1)
        indicator_1 = 1 * (self.Y == 1).reshape(-1, 1)
        m = self.Y.shape[0]

        self._phi = np.sum(indicator_1) / m

        self._mu_0 = np.sum(indicator_0 * self.X, axis=0) / np.sum(indicator_0)
        self._mu_1 = np.sum(indicator_1 * self.X, axis=0) / np.sum(indicator_1)

        self._sig_0 = np.dot(np.transpose(self.X - self._mu_0), indicator_0 * (self.X - self._mu_0)) / np.sum(indicator_0)
        self._sig_1 = np.dot(np.transpose(self.X - self._mu_1), indicator_1 * (self.X - self._mu_1)) / np.sum(indicator_1)

        # with the assumption that sig_0 = sig_1
        self._mu = indicator_0 * self._mu_0 + indicator_1 * self._mu_1
        self._sig = np.dot(np.transpose(self.X - self._mu), self.X - self._mu) / m
            
        if self.boundary.lower() == "linear":
            self.plot_linear()
        elif self.boundary.lower() == "quadratic":
            self.plot_quadratic()
        else:
            self.plot_both()

    def get_quadratic(self):
        x = self.X[:, 0].reshape(-1)
        y = self.X[:, 1].reshape(-1)

        xx, yy = np.mgrid[min(x) - 0.1:max(x)+0.1:.1, min(y)-0.1:max(y)+0.1:.1]
        X = np.c_[xx.ravel(), yy.ravel()]

        sig_0_inv = np.linalg.pinv(self._sig_0)
        sig_1_inv = np.linalg.pinv(self._sig_1)
        
        quadratic_term = np.dot(np.dot(X, sig_1_inv - sig_0_inv), np.transpose(X))

        coefficient_term = -2 * (np.dot(sig_1_inv, self._mu_1) - np.dot(sig_0_inv, self._mu_0))
        linear_term =  np.dot(X, coefficient_term) 

        constant_term = np.dot(np.dot(np.transpose(self._mu_1), sig_1_inv), self._mu_1) - np.dot(np.dot(np.transpose(self._mu_0), sig_0_inv), self._mu_0)
        
        h = np.diag(quadratic_term) + linear_term + constant_term
        h = h.reshape(xx.shape)
        level = np.log(self._phi/(1 - self._phi)) + (np.log(np.linalg.norm(self._sig_0) / np.linalg.norm(self._sig_1)))/2
        level = level / 2

        return xx, yy, h, level

    def get_linear(self):
        x = self.X[:, 0].reshape(-1)
        y = self.X[:, 1].reshape(-1)

        xx, yy = np.mgrid[min(x) - 0.1:max(x)+0.1:.01, min(y)-0.1:max(y)+0.1:.01]
        X = np.c_[xx.ravel(), yy.ravel()]

        sig_inv = np.linalg.pinv(self._sig)
        
        coefficient_term = -2 * np.dot(sig_inv, self._mu_1 - self._mu_0)
        linear_term = np.dot(X, coefficient_term) 

        constant_term = np.dot(np.dot(np.transpose(self._mu_1), sig_inv), self._mu_1) - np.dot(np.dot(np.transpose(self._mu_0), sig_inv), self._mu_0)

        h = linear_term + constant_term
        h = h.reshape(xx.shape)
        level = np.log(self._phi/(1 - self._phi))
        level = level / 2

        return xx, yy, h, level

    def plot_fn(self, fn_list, plot_name, back = True):
        for i in range(len(fn_list)):

            xx, yy, h, level = fn_list[i]()

            if back:
                plt.contourf(xx, yy, h, levels=[-10000, level, 100000], colors = ["green", "red", "blue"], alpha = 0.2)
            plt.contour(xx, yy, h, levels=[level], colors = ["blue"], linewidths = 2)

            x = self.X[:, 0].reshape(-1)
            y = self.X[:, 1].reshape(-1)
            lab_1 = np.asarray([i for i in range(self.Y.shape[0]) if self.Y[i] == 1])
            plt.scatter(x[lab_1], y[lab_1], c = "g", marker="^")
            plt.scatter(x[~lab_1], y[~lab_1], c = "r", marker="x")

        os.makedirs("plots", exist_ok=True)
        # plt.show()
        plt.savefig("plots/" + plot_name, dpi = 300)
        plt.close()
    
    def plot_linear(self):
        self.plot_fn([self.get_linear], "gda_linear.png")
    def plot_quadratic(self):
        self.plot_fn([self.get_quadratic], "gda_quadratic.png")
    def plot_both(self):
        self.plot_fn([self.get_linear, self.get_quadratic], "gda_both.png", back=False)


def main():
    # model = GDA(boundary="linear")
    # model = GDA(boundary="quadratic")
    model = GDA(boundary="both")
    model.load_data("q4x.dat", "q4y.dat")
    model.train()
        
main()