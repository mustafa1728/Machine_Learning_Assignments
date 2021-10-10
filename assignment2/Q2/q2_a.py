import argparse
import pandas as pd
import numpy as np
import cvxopt
from cvxopt import matrix
from libsvm import svmutil
import time 
import json


class SVMClassifier():
    '''
        Main Class for binary classification.
        Implements simple SVM with options for Linear and Gaussian kernels
        Reformulates the SVM dual objective into a quadratic optimisation problem
        and uses CVXOPT for finding optimal solutions.
        Also provides support for using libsvm library.
    '''
    def __init__(self, d=1, C=1, gamma=0.05): # d is the last digit of entry number, i.e. 1
        self.X, self.Y = None, None
        self.d = d
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.eps = 10**(-8)
        self.gamma = gamma
        self.sv_indices = None
        self.bias = 0
        self.result_dict = {}

    def load_data(self, data_path, to_return=False):
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, :-1].values
        X = X / 255
        Y = df.iloc[:, -1].values
        if self.d is not None:
            classes = [self.d, (self.d+1)%10]
            mask = np.asarray([i for i in range(len(Y)) if Y[i] in classes])
            X = X[mask, :]
            Y = Y[mask]
            max_y, min_y = np.max(Y), np.min(Y)
            # This is only to get Y to -1 and 1
            Y = (2*Y - (max_y+min_y)) / (max_y - min_y)
        if to_return:
            return X, Y
        else:
            self.X = X
            self.Y = Y

    def compute_linear_parameters(self):
        Z = np.asarray([self.X[i, :] * self.Y[i] for i in range(len(self.Y))])
        P = matrix(np.dot(Z, np.transpose(Z)), tc='d')
        q = matrix([-1.0 for i in self.Y])

        m = self.Y.shape[0]
        G = np.zeros((2*m, m))
        h = np.zeros((2*m))
        for i in range(m):
            G[i, i] = 1
            h[i] = self.C
            G[i+m, i] = -1
            h[i+m] = 0
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')

        A = matrix(self.Y.reshape(1, -1), tc='d')
        b = matrix([0.0], tc='d')

        return P, q, G, h, A, b
        
    def compute_gaussian_parameters(self):
        Z = np.dot(self.X, np.transpose(self.X))
        diag = Z.diagonal()
        P = diag.reshape(-1, 1) + diag.reshape(1, -1) - 2 * Z
        # P = P * np.dot(self.Y, np.transpose(self.Y))
        P = matrix(np.exp(-1 * self.gamma * P))

        q = matrix([-1.0 for i in self.Y])

        m = self.Y.shape[0]
        G = np.zeros((2*m, m))
        h = np.zeros((2*m))
        for i in range(m):
            G[i, i] = 1
            h[i] = self.C
            G[i+m, i] = -1
            h[i+m] = 0
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')

        A = matrix(self.Y.reshape(1, -1), tc='d')
        b = matrix([0.0], tc='d')

        return P, q, G, h, A, b



    def train(self, kernel="linear"):
        if kernel.lower() == "linear":
            P, q, G, h, A, b = self.compute_linear_parameters()
        elif kernel.lower() == "gaussian":
            P, q, G, h, A, b = self.compute_gaussian_parameters()
        else:
            raise ValueError("Invalid kernel [{}]. Please pass either linear or gaussian".format(kernel))
        sol = cvxopt.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
        self.alpha = np.array(sol["x"])
        self.process_alpha(kernel)

    def process_alpha(self, kernel="linear"):

        self.sv_indices = np.asarray([i for i in range(len(self.alpha)) if np.abs(self.alpha[i]) > self.eps])
        if len(self.sv_indices) > 0:
            self.support_vectors = self.X[self.sv_indices, :]
        else:
            self.support_vectors = np.asarray([[]])
        print("support vectors: ", self.support_vectors.shape)

        if kernel.lower() == "linear":
            w = np.zeros(self.X[0].shape)
            for i in range(len(self.Y)):
                w += self.alpha[i] * self.Y[i] * self.X[i, :]
            
            b_max_term = np.max(np.dot(self.X[self.Y < 0], w))
            b_min_term = np.min(np.dot(self.X[self.Y > 0], w))
            b = -(b_max_term + b_min_term)/2

            self.theta = (w, b)
        else:
            z_sv = np.dot(self.support_vectors, np.transpose(self.support_vectors)).diagonal()
            z_test = np.dot(self.X, np.transpose(self.X)).diagonal()
            P_test = (-2)*np.dot(self.X, np.transpose(self.support_vectors)) + z_sv.reshape(1, -1) + z_test.reshape(-1, 1)
            P_test = np.exp(-1 * self.gamma * P_test)
            preds = np.dot(P_test, self.Y.reshape(-1, 1)[self.sv_indices, :]*self.alpha.reshape(-1, 1)[self.sv_indices, :])

            b_max_term = np.max(preds[self.Y < 0])
            b_min_term = np.min(preds[self.Y >= 0])
            b = -(b_max_term + b_min_term)/2
            self.bias = b

    def test(self, data_path=None, kernel="linear", ret_preds=False, save_result="Q2/results/result_binary.json"):
        if data_path is None:
            X = self.X
            Y = self.Y
        else:
            X, Y = self.load_data(data_path, to_return=True)
        return self.test_data(X, Y, kernel, ret_preds, save_result=save_result)

    def test_data(self, X, Y, kernel="linear", ret_preds=False, save_result=None):
        if kernel.lower() == "linear":
            w, b = self.theta
            preds = np.dot(X, w) + b
        else:
            z_sv = np.dot(self.support_vectors, np.transpose(self.support_vectors)).diagonal()
            z_test = np.dot(X, np.transpose(X)).diagonal()
            P_test = (-2)*np.dot(X, np.transpose(self.support_vectors)) + z_sv.reshape(1, -1) + z_test.reshape(-1, 1)
            P_test = np.exp(-1 * self.gamma * P_test)
            preds = np.dot(P_test, self.Y.reshape(-1, 1)[self.sv_indices, :]*self.alpha.reshape(-1, 1)[self.sv_indices, :])
            # b = np.mean((Y - preds))
            preds = preds + self.bias
        
        no_correct = len([1 for i in range(len(preds)) if preds[i] * Y[i] > 0 or (preds[i]==0 and Y[i] == 1)])
        accuracy = no_correct / len(preds)
        if save_result is not None:
            self.result_dict = {}
            self.result_dict["accuracy"] = "{:.3f}".format(100*accuracy)
            self.result_dict["time"] = "{:.3f}s".format(0)
            preds_list = preds.reshape(-1).tolist()
            preds_list = [1 if p>=0 else -1 for p in preds_list]
            self.result_dict["predictions"] = preds_list
            self.result_dict["labels"] = [int(y) for y in Y]
            with open(save_result, "w") as f:
                json.dump(self.result_dict, f)
        if ret_preds:
            return preds, accuracy
        else:
            return accuracy

    def train_libsvm(self, kernel="linear"):
        if kernel.lower() == "linear":
            self.libsvm_model = svmutil.svm_train(self.Y, self.X, '-c 1 -t 0')
        elif kernel.lower() == "gaussian":
            self.libsvm_model = svmutil.svm_train(self.Y, self.X, '-c 1 -t 2 -g 0.05')
    def test_libsvm(self, data_path=None, kernel="linear"):
        if data_path is None:
            X = self.X
            Y = self.Y
        else:
            X, Y = self.load_data(data_path, to_return=True)
        p_label, p_acc, p_val = svmutil.svm_predict(Y, X, self.libsvm_model)
        return p_acc

def get_args():
    # for getting command line arguments
    parser = argparse.ArgumentParser(description='Gradient Descent')
    parser.add_argument('--train_path', type=str, default="./data/Music_Review_train.json", help='the file path where training data is stored')
    parser.add_argument('--test_path', type=str, default="./data/Music_Review_test.jso", help='the file path where test data is stored')
    parser.add_argument('--part', type=str, default="a", help='the part to be run')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    print("Running Part {} of question 2 binary".format(args.part.lower()))

    ### Part a and b
    if args.part.lower() == "a" or args.part.lower() == "b":
        ### Part a
        if args.part.lower() == "a":
            kernel="linear"
        ### Part b
        else:
            kernel="gaussian"
        model = SVMClassifier()
        model.load_data(args.train_path)
        start_time = time.time()
        model.train(kernel=kernel)
        print("Time for CVXOPT: ", time.time() - start_time)
        print("Training Accuracy: ", model.test(kernel=kernel))
        print("Test Accuracy: ", model.test(args.test_path, kernel=kernel))
    
    ### Part c
    elif args.part.lower() == "c":
        for kernel in ["linear", "gaussian"]:
            start_time = time.time()
            model = SVMClassifier()
            model.load_data(args.train_path)
            model.train_libsvm(kernel=kernel)
            print("Time for libsvm: ", time.time() - start_time)
            print("Training Accuracy ({}): ".format(kernel), model.test_libsvm(kernel=kernel))
            print("Test Accuracy ({}): ".format(kernel), model.test_libsvm(args.test_path, kernel=kernel))

    else:
        raise ValueError("The part should be one of a-c, got {}.".format(args.part))
    


if __name__ == "__main__":
    main()