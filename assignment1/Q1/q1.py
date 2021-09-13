import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import os

class GradientDescent():
    '''
        Class Implementing Gradient Descent
        Q1 of COL774 Assignment1

        Part a) Implementation
            The class is intialised with learning rate or step size
            Then data is loaded using model.load_data(x_path, y_path)
            Then training starts with model.train()
            The main parameter update step is in training_step()
            Stopping criteria is in stopping()
            train() drives the training by calling training_step() in each iteration, and saves the parameter and loss history
        
        Part b) Data and Hypothesis
            Implemented in _plot_data_hyp()
            Also included in the animation for subsequent parts for better visualisation

        Part c) 3D loss
            Implemented in _plot_loss_3d()

        Part d) Contour Loss
            Implemented in _plot_loss_contours
            This and the plots from previous 2 parts are animated in visualise()
            The plots are also saved individually and combined.

        Part e) Repeat for different step sizes
            The code is run 3 times with the different step sizes as arguments
    '''
    def __init__(self, lr):
        self.lr = lr
        self.X = None
        self.Y = None
        self._theta = None
        self._grad = None
        self._loss_history = []
        self._theta_history = []
        self._no_iterations = 0
    
    def summary(self):
        print("Learning Rate {}".format(self.lr))
        print("Stopping criteria: Loss difference <= 10^(-6) Loss")
        if self._no_iterations > 0:
            print("=========================================================")
            print("Total {} iterations".format(self._no_iterations))
            print("Final parameters: theta {}".format(self._theta.reshape(-1)))
            print("Final loss is {}".format(self.loss()))


    def load_data(self, x_path, y_path):
        # For loading the data used to train the model
        self.X = pd.read_csv(x_path).values
        self.Y = pd.read_csv(y_path).values
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        self.X = np.append(np.ones((self.X.shape[0], 1)), self.X.reshape(-1, 1), axis = 1)

    def training_step(self, data):
        (X, Y) = data
        m = Y.shape[0]
        # computing gradient
        self._grad = - np.dot(np.transpose(X), (Y -  np.dot(X, self._theta))) / m
        self._grad = self._grad.reshape(-1, 1)
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
        self._loss_history.append(self.loss())
        self._theta_history.append(self._theta)

        for iter in range(max_iter):
            if self._grad is not None and self.stopping():
                break
            self.training_step((self.X, self.Y))
            self._loss_history.append(self.loss())
            self._theta_history.append(self._theta)
            self._no_iterations += 1

    def stopping(self):
        # The stopping criteria used
        loss_difference = abs(self._loss_history[-1] - self._loss_history[-2])
        threshold = 10**(-6) * abs(self._loss_history[-1])
        return loss_difference <= threshold

    def loss(self): 
        # Mean squared loss
        Y_hat = np.dot(self.X, self._theta).reshape(-1)
        squares = np.square(self.Y -  Y_hat)
        loss = np.mean(squares) / 2
        return loss

    def _loss_for_plot(self, theta0, theta1):
        # same loss for above, but makes it easier with input specifications used in plotting
        no_samples = self.Y.shape[0]
        diffs = [self.Y[i] - theta0 - theta1*self.X[i, 1] for i in range(no_samples)]
        squares = [i**2 for i in diffs]
        loss = sum(squares) / (2 * no_samples)
        return loss

    def _plot_data_hyp(self, theta_idx = -1):
        # Plotting the normalised data and learned hypothesis
        x = self.X[:, 1]
        y = self.Y
        # plotting normalised data
        self.ax_hyp.scatter(x, y, label = "Normalised Data")
        arg_s = np.argsort(x)
        new_x = self.X[arg_s, 1]
        theta = self._theta_history[theta_idx]
        Y_hypth = np.dot(self.X[arg_s], theta)
        # plotting Learned hypothesis
        self.ax_hyp.plot(new_x, Y_hypth, color='red', label = "Learned Hypothesis")
        self.ax_hyp.set_xlabel("Normalised Acidity of Wine")
        self.ax_hyp.set_ylabel("Density of Wine")
        self.ax_hyp.set_title("Data And Learned Hypothesis")
        self.ax_hyp.legend()

    
    def _plot_loss_3d(self, indices_list):
        
        theta0 = np.asarray([i[0] for i in self._theta_history]).reshape(-1)
        theta1 = np.asarray([i[1] for i in self._theta_history]).reshape(-1)
        if indices_list is not None:
            theta0 = [theta0[i] for i in indices_list]
            theta1 = [theta1[i] for i in indices_list]
            hist = [self._loss_history[i] for i in indices_list]
        # this plots the movement of parameters theta
        self.ax_3d.plot(theta0, theta1, hist, color = "red", zorder=10)
        # Here, just the initial and final positions are plotted
        size = 0.001
        line_initial = ( [theta0[0] - size, theta0[0] + size] , [theta1[0] - size, theta1[0] + size], [hist[0] - size, hist[0] + size])
        self.ax_3d.plot(line_initial[0], line_initial[1], line_initial[2], color = "blue", zorder=11, label="Initial Theta", linewidth = 5)
        line_final = ( [theta0[-1] - size, theta0[-1] + size] , [theta1[-1] - size, theta1[-1] + size], [hist[-1] - size, hist[-1] + size])
        self.ax_3d.plot(line_final[0], line_final[1], line_final[2], color = "green", zorder=12, label="Final Theta", linewidth = 5)
        self.ax_3d.legend()

        # Uniform samples from set interval. Interval chosen after running once, looking at converged parameters, and then deciding
        theta0 = np.linspace(-0.2, 1.8, 30)
        theta1 = np.linspace(-1, 1, 30)
        theta0, theta1 = np.meshgrid(theta0, theta1)
        J = self._loss_for_plot(theta0, theta1)
        
        # The surface of the loss function is plotted here
        self.ax_3d.plot_surface(theta0, theta1, J, cmap='viridis', alpha=1, zorder=0)
        self.ax_3d.set_title('Loss vs parameters')
        self.ax_3d.set_xlabel("Theta 0")
        self.ax_3d.set_ylabel("Theta 1")
        self.ax_3d.set_zlabel("Mean square loss")
        # plt.show()
    
    def _plot_loss_contours(self, indices_list, mode = "line"):
        # same set of theta values taken as in th3 3d plot for consistency.
        theta0 = np.linspace(-0.2, 1.8, 30)
        theta1 = np.linspace(-1, 1, 30)
        theta0, theta1 = np.meshgrid(theta0, theta1)
        J = self._loss_for_plot(theta0, theta1)

        # Contour plotting
        cp = self.ax_cnt.contourf(theta0, theta1, J)
        self.ax_cnt.set_title('Loss vs parameters')
        self.ax_cnt.set_xlabel("Theta 0")
        self.ax_cnt.set_ylabel("Theta 1")

        theta0 = [i[0] for i in self._theta_history]
        theta1 = [i[1] for i in self._theta_history]
        # Plot the theta values
        # Depending on mode, either use a scatter plot or line plot
        if indices_list is not None:
            theta0 = [theta0[i] for i in indices_list]
            theta1 = [theta1[i] for i in indices_list]
        if mode == "line":
            self.ax_cnt.plot(theta0, theta1, color='red')
        else:
            self.ax_cnt.scatter(theta0, theta1, color='red', s = 2)
        # Initial and final points
        self.ax_cnt.scatter(theta0[0], theta1[0], color='blue', zorder = 10, label="Initial Theta")
        self.ax_cnt.scatter(theta0[-1], theta1[-1], color='green', zorder = 10, label="Final Theta")
        self.ax_cnt.legend()
    
    def save_plots(self, suffix = "", save_dir = "plots"):
        # used to save last versions of the plots obtained
        if suffix is None:
            suffix = ""
        os.makedirs(save_dir, exist_ok = True)
        self.fig.savefig(os.path.join(save_dir, "combined.pdf"), dpi = 80)

        extent = self.ax_3d.get_tightbbox(self.fig.canvas.get_renderer()).transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(os.path.join(save_dir, '3d_loss.pdf'), bbox_inches=extent, dpi = 300)

        extent = self.ax_hyp.get_tightbbox(self.fig.canvas.get_renderer()).transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(os.path.join(save_dir, 'hypothesis.pdf'), bbox_inches=extent, dpi = 300)

        extent = self.ax_cnt.get_tightbbox(self.fig.canvas.get_renderer()).transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(os.path.join(save_dir, 'contour_loss.pdf'), bbox_inches=extent, dpi = 300)
    
    def _plot_animate(self, i):
        # Single animation step for visualising Gradient descent process
        # First all axes are cleared
        self.ax_3d.clear()
        self.ax_hyp.clear()
        self.ax_cnt.clear()
        indices_list = [k for k in range(i+1)]
        # partial data upto current position is plotted
        self._plot_data_hyp(i)
        self._plot_loss_3d(indices_list)
        self._plot_loss_contours(indices_list)

    def visualise(self, save_plots=False, save_anim=False, anim_iterations = None, save_dir = "plots"):
        # Unified function that calls all individual plots and animates them
        self.fig = plt.figure(constrained_layout=True, figsize=(12, 7))
        gs = GridSpec(2, 5, figure=self.fig)
        self.ax_3d = self.fig.add_subplot(gs[:, :3], projection='3d')
        self.ax_3d.view_init(elev=45, azim=21)
        self.ax_hyp = self.fig.add_subplot(gs[0, 3:]) 
        self.ax_cnt = self.fig.add_subplot(gs[1, 3:])
        if anim_iterations is None:
            anim_iterations = len(self._loss_history) - 1
        ani = animation.FuncAnimation(self.fig, self._plot_animate, interval=1, frames=anim_iterations, repeat=False) 
        plt.show()
        if save_plots:
            self.save_plots(anim_iterations, save_dir = save_dir)
        if save_anim:
            os.makedirs(save_dir, exist_ok=True)
            ani.save(os.path.join(save_dir, 'gradient_descent.gif'), writer='imagemagick') 

    def plot_loss_history(self):
        # for plotting loss vs no of iterations
        plt.ylim(-0.05, 0.55)
        iterations = [i for i in range(len(self._loss_history))]
        plt.plot(iterations, self._loss_history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Mean Squared Loss")
        plt.title("Loss vs Number of Iterations")
        plt.show()
        plt.savefig("plots/loss_it_lr{}.pdf".format(self.lr), dpi = 300)

def get_args():
    # for getting command line arguments
    parser = argparse.ArgumentParser(description='Gradient Descent')
    parser.add_argument('--data_path', type=str, default=".", help='the directory path where data files are stored')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate or step size')
    parser.add_argument('--save_path', type=str, default="plots", help='the directory path where plots will be saved')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    model = GradientDescent(lr = args.lr)
    model.load_data(os.path.join(args.data_path, "linearX.csv"), os.path.join(args.data_path, "linearY.csv"))
    model.train()
    model.summary()
    model.visualise(save_plots = True, save_anim=False, anim_iterations = None)
    model.plot_loss_history()
    
    

if __name__ == "__main__":
    main()
