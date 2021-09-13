import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


class GDA():
    '''
        Gaussian Discriminant Analysis
        Q4 of COL774 Assignment1

        Part a) Special condition sigmas assumed equal
            The class is intialised without any parameters
            Then data is loaded using model.load_data(x_path, y_path) and normalised
            The main part is train() function where all the parameter values are obtained for both linear and quadratic boundaries
            The values of means and sigma or sigmas are all calculated in the train() function.
        
        Part b) Plot Training Data 
            The data is plot using the general purpose plot_fn() function
            Passing no functions in its function list leads to a plot having only the data
            The plot includes crosses and traingles and also different colors representing the different labels.

        Part c) Linear boundary Equation
            Implemented in get_linear()
            The equations used follow the formulae derived in class
            For more detailed explanation, refer to the report

        Part d) General case
            The parameter obtained in this case are also computed in the train() function as well
            Again, the formulae concerned were derived in class, and implementations have been explained in the report
            All values of means and variances and phi are calculated and reported

        Part e) Quadratic boundary Equation
            Implemented in get_quadratic()
            The equations used follow the formulae derived in class
            For more detailed explanation, refer to the report
            A separate plot as well as both plots combined is drawn for better clarity
        
        Part f) Comments
            Discussed in the report
    '''
    def __init__(self):
        pass

    def summary(self):
        print("Values of Parameters obtained by the closed form solution for Gaussian Discriminant Analysis.")
        print("Phi: {}".format(self._phi))
        print("Means: {}, {}".format(self._mu_0, self._mu_1))
        print("Sigma: {}".format(self._sig.tolist()))
        print("Sigma_0: {}".format(self._sig_0.tolist()))
        print("Sigma_1: {}".format(self._sig_1.tolist()))

    def load_data(self, x_path, y_path):
        self.X = np.genfromtxt(x_path, dtype= (int,int))
        self.Y = np.genfromtxt(y_path, dtype = "str")
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
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

    def get_quadratic(self):
        # To get the quadratic boundary 
        x = self.X[:, 0].reshape(-1)
        y = self.X[:, 1].reshape(-1)

        xx, yy = np.mgrid[min(x) - 0.1:max(x)+0.1:.1, min(y)-0.1:max(y)+0.1:.1]
        X = np.c_[xx.ravel(), yy.ravel()]

        sig_0_inv = np.linalg.pinv(self._sig_0)
        sig_1_inv = np.linalg.pinv(self._sig_1)
        
        quadratic_term = (1/2) * np.dot(np.dot(X, sig_1_inv - sig_0_inv), np.transpose(X))

        coefficient_term = -1 * (np.dot(sig_1_inv, self._mu_1) - np.dot(sig_0_inv, self._mu_0))
        linear_term =  np.dot(X, coefficient_term) 

        constant_term_1 = np.dot(np.dot(np.transpose(self._mu_1), sig_1_inv), self._mu_1)
        constant_term_2 = np.dot(np.dot(np.transpose(self._mu_0), sig_0_inv), self._mu_0)
        constant_term_3 = np.log((1 - self._phi)/self._phi)
        constant_term_4 = (np.log(np.linalg.norm(self._sig_1) / np.linalg.norm(self._sig_0)))/2

        constant_term =  (1/2) * (constant_term_1 - constant_term_2) + constant_term_3 + constant_term_4
        
        # follows the formulae discussed in class
        # diag is needed, else dimensions inconsistent
        h = np.diag(quadratic_term) + linear_term + constant_term
        h = h.reshape(xx.shape)
        level = 0

        return xx, yy, h, level

    def get_linear(self):
        # To get the linear boundary 
        x = self.X[:, 0].reshape(-1)
        y = self.X[:, 1].reshape(-1)

        xx, yy = np.mgrid[min(x) - 0.1:max(x)+0.1:.01, min(y)-0.1:max(y)+0.1:.01]
        X = np.c_[xx.ravel(), yy.ravel()]

        sig_inv = np.linalg.pinv(self._sig)
        
        coefficient_term = -1 * np.dot(np.transpose(self._mu_1 - self._mu_0), sig_inv)
        linear_term = np.dot(X, coefficient_term) 

        constant_term_1 = np.dot(np.dot(np.transpose(self._mu_1), sig_inv), self._mu_1) 
        constant_term_2 = np.dot(np.dot(np.transpose(self._mu_0), sig_inv), self._mu_0)
        constant_term_3 = np.log((1 - self._phi)/self._phi)
        constant_term = (constant_term_1 - constant_term_2) / 2 + constant_term_3

        # follows the formulae discussed in class
        h = linear_term + constant_term
        h = h.reshape(xx.shape)
        level = 0
        level = level 

        return xx, yy, h, level

    def plot_fn(self, fn_list, plot_name, plot_title = "", back = True, save_dir = "plots"):
        # Generic plotting function
        # takes as input a list of functions (callables), where each function generates the data to plot
        # All 4 plots (data, linear, quadratic and both) are plotted using this same function, by passing different values in fn_list
        for i in range(len(fn_list)):
            xx, yy, h, level = fn_list[i]()
            if back:
                plt.contourf(xx, yy, h, levels=[-10000, level, 100000], colors = ["green", "red", "blue"], alpha = 0.2)
            plt.contour(xx, yy, h, levels=[level], colors = ["blue"], linewidths = 2)

        x = self.X[:, 0].reshape(-1)
        y = self.X[:, 1].reshape(-1)
        lab_1 = np.asarray([i for i in range(self.Y.shape[0]) if self.Y[i] == 1])
        plt.scatter(x[lab_1], y[lab_1], c = "g", marker="^", label = "Canada")
        plt.scatter(x[~lab_1], y[~lab_1], c = "r", marker="x", label = "Alaska")
        plt.xlabel("Normalised growth ring diameters in fresh water")
        plt.ylabel("Normalised growth ring diameters in marine water")
        plt.title(plot_title)
        plt.legend()

        os.makedirs(save_dir, exist_ok=True)
        # plt.show()
        plt.savefig(os.path.join(save_dir, plot_name), dpi = 300)
        plt.close()
    
    def plot_linear(self, save_dir):
        self.plot_fn([self.get_linear], "gda_linear.pdf", plot_title="Linear Decision Boundary", save_dir=save_dir)
    def plot_quadratic(self, save_dir):
        self.plot_fn([self.get_quadratic], "gda_quadratic.pdf", plot_title="Quadratic Decision Boundary", save_dir=save_dir)
    def plot_both(self, save_dir):
        self.plot_fn([self.get_linear, self.get_quadratic], "gda_both.pdf", plot_title="Both Linear and Quadratic Decision Boundaries", back=False, save_dir=save_dir)

    def visualise(self, boundary, save_dir = "plots"):
        # visualising the different plots
        if boundary.lower() == "none":
            self.plot_fn([], "gda_data.pdf", plot_title = "Normalised Data")
        elif boundary.lower() == "linear":
            self.plot_linear(save_dir)
        elif boundary.lower() == "quadratic":
            self.plot_quadratic(save_dir)
        else:
            self.plot_both(save_dir)

def get_args():
    # for getting command line arguments
    parser = argparse.ArgumentParser(description='Gaussian Discriminant Analysis')
    parser.add_argument('--data_path', type=str, default=".", help='the directory path where data files are stored')
    parser.add_argument('--save_path', type=str, default="plots", help='the directory path where plots will be saved')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    model = GDA()
    model.load_data(os.path.join(args.data_path, "q4x.dat"), os.path.join(args.data_path, "q4y.dat"))
    model.train()
    model.summary()
    # Part b)
    model.visualise(boundary="none", save_dir = args.save_path)
    # Part c)
    model.visualise(boundary="linear", save_dir = args.save_path)
    # Part e)
    model.visualise(boundary="quadratic", save_dir = args.save_path)
    model.visualise(boundary="both", save_dir = args.save_path)
    
        
main()