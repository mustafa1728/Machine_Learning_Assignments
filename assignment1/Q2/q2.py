import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time

class NormalDistribution():
    '''
        Normal Distribution
        Q2 Part a) of COL774 Assignment1

        Separate class for ease of use and intuitive interface
        use sample() to get a single sample from the distribution
        use sample(no_samples) to get an array of no_samples
    '''
    def __init__(self, mean, var, seed = 0):
        self.mu = mean
        self.sigma = np.sqrt(var)
        np.random.seed(seed)

    def sample(self, no_samples=None):
        if no_samples is None:
            return np.random.normal(self.mu, self.sigma)
        else:
            return np.random.normal(self.mu, self.sigma, no_samples)

def sample(no_samples, theta):
    # Getting the million samples
    x1 = NormalDistribution(3, 4)
    x2 = NormalDistribution(-1, 4)
    noise = NormalDistribution(0, 2)

    X = np.asarray([[1, x1.sample(), x2.sample()] for i in range(no_samples)])
    noise_samples = np.asarray([noise.sample() for i in range(no_samples)])
    Y = np.dot(X, theta) + noise_samples.reshape(-1, 1)

    return X, Y




class StochasticGradientDescent():
    '''
        Stochastic Gradient Descent
        Q2 Part b), c), d) of COL774 Assignment1

        Part b) Implementation
            The implementation is very similar to Q1
            No change in training_step(), only train() changed 
            Stopping critera does not take only the last 2 step, but average of last k steps (more details in report)
            Data is shuffled in each iteration before starting the first batch
            Different batch sizes are looped over in the main() function
            Including million batch size is optional, since it takes a lot of time to converge and when included, the iterations are capped at 100
            Stopping criteria are implemented in stopping(), and the exact criteria is dynamically chosen using the batch size 

        Part c) Comparision for different batch sizes and test loss
            The loss function is modified to include option to calculate on external data
            This same is used to get the test loss, by passing the test data as input
            The same loss function is further modified to take parameters as input.
            This is used to get the loss for the original hypothesis by hard coding its value

        Part d) Plot movement of theta in 3D space
            The function plot_theta() is used to implement this.
            In addition to the 3D plot, a loss vs no. of iterations plot is also made
            Since three different plots are made, 
    '''
    def __init__(self, lr, batch_size, seed = 0):
        self.lr = lr
        self.X = None
        self.Y = None
        self._theta = None
        self._grad = None
        self._loss_history = []
        self._theta_history = []
        self.batch_size = batch_size
        np.random.seed(seed)
    
    def set_data(self, X, Y):            
        self.X = X
        self.Y = Y

    def shuffle(self):
        # randomly shuffles the data
        idx = np.arange(self.Y.shape[0])
        np.random.shuffle(idx)
        self.X = self.X[idx, :]
        self.Y = self.Y[idx, :]

    def stopping(self):
        no_examples = 10000
        if len(self._loss_history) > 2*no_examples:
            diff_avg = np.mean(np.asarray(self._loss_history[-no_examples:])) - np.mean(np.asarray(self._loss_history[-2*no_examples:-no_examples]))
            if np.abs(diff_avg) <= 10**(-5):
                return True
        return False

    def loss(self, X=None, Y=None, theta=None): 
        # Same Mean squared loss, modified to take external data, or pretrained weights as input
        # Use discussed better in class docstrings
        if X is None or Y is None:
            X = self.X
            Y = self.Y
        if theta is None:
            theta = self._theta
        Y_hat = np.dot(X, theta).reshape(-1)
        Y = Y.reshape(-1)
        squares = np.square(Y -  Y_hat)
        loss = np.mean(squares) / 2
        return loss

    def training_step(self, data):
        (X, Y) = data
        m = Y.shape[0]
        # computing gradient
        self._grad = - np.dot(np.transpose(X), (Y -  np.dot(X, self._theta))) / m
        self._grad = self._grad.reshape(-1, 1)
        # optimisation step
        self._theta = self._theta - self.lr * self._grad
    
    def train(self, max_iter = 1000000, display_iter = 1000, verbose = 0):
        # Initialising weights (X already has intercept term)
        self._theta = np.zeros((self.X.shape[1], 1))
        if verbose:
            print("Intial Loss is {}".format(self.loss()))
        self.no_batches = self.X.shape[0] // self.batch_size
        self.no_steps = 0
        self.no_iterations = 0
        for iter in range(max_iter):
            self.no_iterations += 1
            self.shuffle()
            for batch in range(self.no_batches):
                self.no_steps += 1
                x_batch = self.X[batch * self.batch_size : (batch + 1) * self.batch_size]
                y_batch = self.Y[batch * self.batch_size : (batch + 1) * self.batch_size]
                self.training_step((x_batch, y_batch))

                self._loss_history.append(self.loss(x_batch, y_batch))
                self._theta_history.append(self._theta)
                if self._grad is not None and self.stopping():
                    break
            if verbose == 2 and iter % display_iter == 0:
                print("Iteration {} | Loss {}".format(iter+1, self.loss()))
            if self._grad is not None and self.stopping():
                break
        if verbose:
            print("Total {} iterations. Final parameters: theta {}".format(iter+1, self._theta.reshape(-1)))
            print("Final loss is {}".format(self.loss()))

    def plot_theta(self, save_dir = "plots"):
        # for plotting 3D plot of movement of theta
        fig = plt.figure()
        self.ax_3d = fig.add_subplot(projection='3d')
        theta0 = np.asarray([i[0] for i in self._theta_history])
        theta1 = np.asarray([i[1] for i in self._theta_history])
        theta2 = np.asarray([i[2] for i in self._theta_history])
        self.ax_3d.plot(theta0.reshape(-1), theta1.reshape(-1), theta2.reshape(-1), color = "red", linewidth = 1)
        
        extent = self.ax_3d.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        self.ax_3d.set_title('Movement of parameters')
        self.ax_3d.set_xlabel("Theta 0")
        self.ax_3d.set_ylabel("Theta 1")
        self.ax_3d.set_zlabel("Theta 2")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "sgd_batch_{}.pdf".format(self.batch_size)), dpi = 300)
        plt.close('all')

    def plot_loss_history(self, save_dir = "plots"):
        # For plotting loss vs no of steps
        iterations = [i for i in range(len(self._loss_history))]
        plt.plot(iterations, self._loss_history)
        plt.xlabel("Number of Training Steps")
        plt.ylabel("Mean Squared Loss")
        plt.title("Loss vs Number of Training Steps")
        plt.savefig(os.path.join(save_dir, "loss_it_batch{}.pdf".format(self.batch_size)), dpi = 300)

def get_args():
    # for getting command line arguments
    parser = argparse.ArgumentParser(description='Stochastic Gradient Descent')
    parser.add_argument('--data_path', type=str, default=".", help='the directory path where test data files are stored')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate or step size')
    parser.add_argument('--save_path', type=str, default="plots", help='the directory path where plots will be saved')
    parser.add_argument('--include_mil', type=bool, default=False, help='whether to include batch size million or not')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    orig_theta = np.asarray([[3], [1], [2]])
    X, Y = sample(1000000, orig_theta)
    separator = "================================================================================================"
    single_separator = separator.replace("=", "-")
    print(separator)
    print("Question 2 part a\n")
    print("X | shape: {} mean: {:>40} std: {:>40}".format(X.shape, str(np.mean(X, axis = 0)), str(np.std(X, axis = 0))))
    print("Y | shape: {} mean: {:>40} std: {:>40}".format(Y.shape, str(np.mean(Y, axis = 0)), str(np.std(Y, axis = 0))))
    batch_sizes = [1, 100, 10000]
    if args.include_mil:
        batch_sizes.append(1000000)

    models = [StochasticGradientDescent(lr = args.lr, batch_size=b) for b in batch_sizes]
    times = []
    # Comment this out to run million batch size to convergence as well
    if args.include_mil:
        max_iter = 100
    else:
        max_iter = 1000000
    for model in models:
        model.set_data(X, Y)
        start_time = time.time()
        model.train(max_iter=max_iter)
        times.append(time.time() - start_time)

    print(separator)
    print("Question 2 part b\n")
    print("Learned Parameters")
    for i in range(len(models)):
        print("Batch size {: >8} | {}".format(batch_sizes[i], models[i]._theta.reshape(-1)))

    print(separator)
    print("Question 2 part c\n")
    print("Distance between the parameters learned and original parameters (Norm)")
    for i in range(len(models)):
        print("Batch size {: >8} | {:>10.5f}".format(batch_sizes[i], float(np.linalg.norm(models[i]._theta - orig_theta))))
    
    print(single_separator)
    print("Time taken for training")
    for i in range(len(models)):
        print("Batch size {: >8} | {:>10.3f}s".format(batch_sizes[i], times[i]))

    print(single_separator)
    print("Number of iterations and steps for training")
    for i in range(len(models)):
        print("Batch size {: >8} | {: >8} {: >8}".format(batch_sizes[i], models[i].no_iterations, models[i].no_steps))

    print(single_separator)
    print("Losses on the train set")
    model_temp = StochasticGradientDescent(lr = 0.001, batch_size=1)
    for i in range(len(models)):
        print("Batch size {: >8} | {:>10.5f}".format(batch_sizes[i], models[i].loss()))
    model_temp = StochasticGradientDescent(lr = 0.001, batch_size=1)
    print("Original Hypothesis | {:>10.7f}".format(model_temp.loss(X, Y, theta=orig_theta)))
    
    print(single_separator)
    test_data = pd.read_csv(os.path.join(args.data_path, "q2test.csv"))
    X_test = test_data.iloc[:, :2].values
    X_test= np.append(np.ones((X_test.shape[0], 1)), X_test, axis = 1)
    Y_test = test_data.iloc[:, 2].values
    
    print("Losses on the test set")
    for i in range(len(models)):
        print("Batch size {: >8} | {:>10.5f}".format(batch_sizes[i], models[i].loss(X_test, Y_test)))
    
    print("Original Hypothesis | {:>10.7f}".format(model_temp.loss(X_test, Y_test, orig_theta)))
    
    print(separator)
    print("Question 2 part d\n")
    for model in models:
        model.plot_theta(save_dir = args.save_path)
        model.plot_loss_history(save_dir = args.save_path)
    print(separator)


main()