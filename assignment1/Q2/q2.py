from Q1.q1 import GradientDescent

import os
import numpy as np
import matplotlib.pyplot as plt
import time

class NormalDistribution():
    def __init__(self, mean, var, seed = 0):
        self.mu = mean
        self.sigma = np.sqrt(var)
        np.random.seed(seed)

    def sample(self):
        return np.random.normal(self.mu, self.sigma)

def sample(no_samples):
    x1 = NormalDistribution(3, 4)
    x2 = NormalDistribution(-1, 4)
    noise = NormalDistribution(0, 2)
    theta = np.asarray([[3], [1], [2]])

    X = np.asarray([[1, x1.sample(), x2.sample()] for i in range(no_samples)])
    noise_samples = np.asarray([noise.sample() for i in range(no_samples)])
    Y = np.dot(X, theta) + noise_samples.reshape(-1, 1)

    return X, Y




class StochasticGradientDescent(GradientDescent):
    def __init__(self, lr, batch_size, seed = 0):
        super().__init__(lr)
        self.batch_size = batch_size
        np.random.seed(seed)
    
    def set_data(self, X, Y):            
        self.X = X
        self.Y = Y

    def shuffle(self):
        idx = np.arange(self.Y.shape[0])
        np.random.shuffle(idx)
        self.X = self.X[idx, :]
        self.Y = self.Y[idx, :]
    
    def train(self, max_iter = 1000000, display_iter = 1000, verbose = True):
        self.check_sanity()

        # Initialising weights (X already has intercept term)
        self._theta = np.zeros((self.X.shape[1], 1))
        # self._theta = np.asarray([[3], [1], [2]])
        if verbose:
            print("Intial Loss is {}".format(self.mean_squared_loss()))
        no_batches = self.X.shape[0] // self.batch_size
        for iter in range(max_iter):
            self.shuffle()
            for batch in range(no_batches):
                x_batch = self.X[batch * self.batch_size : (batch + 1) * self.batch_size]
                y_batch = self.Y[batch * self.batch_size : (batch + 1) * self.batch_size]
                self.training_step((x_batch, y_batch))

                # print("grad: {}, theta: {}, loss: {}".format(self._grad.reshape(-1), self._theta.reshape(-1), self.mean_squared_loss()))
                self._loss_history.append(self.mean_squared_loss())
                self._theta_history.append(self._theta)
            if verbose == 2 and iter % display_iter == 0:
                print("Iteration {} | Loss {}".format(iter+1, self.mean_squared_loss()))
            if self._grad is not None and self.stopping():
                break
        if verbose:
            print("Total {} iterations. Final parameters: theta {}".format(iter+1, self._theta.reshape(-1)))
            print("Final loss is {}".format(self.mean_squared_loss()))

    def plot_theta(self):
        self.ax_3d = plt.axes(projection='3d')
        theta0 = np.asarray([i[0] for i in self._theta_history])
        theta1 = np.asarray([i[1] for i in self._theta_history])
        theta2 = np.asarray([i[2] for i in self._theta_history])
        self.ax_3d.plot(theta0.reshape(-1), theta1.reshape(-1), theta2.reshape(-1), color = "red", linewidth = 2)
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/sgd_loss_batch_{}.png".format(self.batch_size))
        # plt.show()
        plt.close('all')


def main():
    X, Y = sample(1000000)
    b = 1
    model = StochasticGradientDescent(lr = 0.0000001, batch_size=b)
    model.set_data(X, Y)
    start_time = time.time()
    model.train(display_iter = 1, verbose=1, max_iter=10)
    model.plot_theta()
    print("Training with batch size {} took {} seconds".format(b, time.time() - start_time))
    # for b in [1000000, 10000, 100, 1]:
    #     print("Running SGD with batch size {}".format(b))
        
    #     print("Training with batch size {} took {} seconds".format(b, time.time() - start_time))
    #     model.plot_theta()

main()