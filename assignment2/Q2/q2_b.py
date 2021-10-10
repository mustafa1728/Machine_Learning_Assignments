import argparse
import os
import cv2
import pandas as pd
import numpy as np
from q2_a import SVMClassifier
import json
import time
from libsvm import svmutil
import matplotlib.pyplot as plt
import seaborn as sns

class SVMMultiClassClassifier():
    '''
        Main Class for multi class classification.
        Implements one vs one classification.
        Imports the Binary classifier from the previous part, and make kC2
        instances of the classifier. 
        The training data is split and preprocessed before passing 
        to the required classifier.
        Then, the results from each classifier are again processed to 
        obtain final voted classification prediction.
    '''
    def __init__(self, C=1, gamma=0.05): 
        self.X, self.Y = None, None
        self.C = C
        self.alpha = None
        self.gamma = gamma
        self.classifiers = None
        self.classifier_indices = []
        self.result_dict = {}

    def set_c(self, c):
        self.C = c

    def load_data(self, data_path, to_return=False):
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, :-1].values
        X = X / 255
        Y = df.iloc[:, -1].values
        if to_return:
            return X, Y
        else:
            self.X = X
            self.Y = Y

    def init_classifiers(self):
        classes = list(set(self.Y))
        no_classes = len(classes)
        self.classifiers = [[SVMClassifier() for i in range(no_classes)] for j in range(no_classes)]
        self.classifier_indices = [[None for i in range(no_classes)] for j in range(no_classes)]

    def get_subset(self, X, Y, classes):
        mask = np.asarray([i for i in range(len(Y)) if Y[i] in classes])
        X = X[mask, :]
        Y = Y[mask]
        max_y, min_y = np.max(Y), np.min(Y)
        # This is only to get Y to -1 and 1
        Y = (2*Y - (max_y+min_y)) / (max_y - min_y)
        return X, Y, mask

    def set_classifiers_data(self):
        classes = list(set(self.Y))
        no_classes = len(classes)
        for i in range(no_classes):
            for j in range(i+1, no_classes):
                class_i = classes[i]
                class_j = classes[j]
                temp_X, temp_Y, indices = self.get_subset(self.X, self.Y, classes = [class_i, class_j])
                self.classifiers[i][j].X = temp_X
                self.classifiers[i][j].Y = temp_Y
                self.classifier_indices[i][j] = indices
    
    def train(self, kernel="gaussian"):
        self.train_time = time.time()
        classes = list(set(self.Y))
        no_classes = len(classes)
        for i in range(no_classes):
            for j in range(i+1, no_classes):
                print("Training to classify between class {} and class {}".format(classes[i], classes[j]))
                self.classifiers[i][j].train(kernel)
        self.train_time = time.time() - self.train_time

    def test(self, data_path=None, kernel="gaussian", result_save="result.json"):
        classes = list(set(self.Y))
        no_classes = len(classes)
        if data_path is None:
            X = self.X
            Y = self.Y
        else:
            X, Y = self.load_data(data_path, to_return=True)
        
        all_preds = np.asarray([[-np.ones(Y.shape) for i in range(no_classes)] for j in range(no_classes)])
        all_scores = np.asarray([[-np.ones(Y.shape) for i in range(no_classes)] for j in range(no_classes)])
        for i in range(no_classes):
            for j in range(i+1, no_classes):
                temp_X, temp_Y, indices = self.get_subset(X, Y, classes = [classes[i], classes[j]])
                preds, _ = self.classifiers[i][j].test_data(temp_X, temp_Y, kernel=kernel, ret_preds=True)

                all_scores[i][j][indices] = np.asarray(preds).reshape(-1)
                preds = [j if p>=0 else i for p in preds]
                all_preds[i][j][indices] = np.asarray(preds).reshape(-1).astype("uint8")
        
        frequencies = [{cls:0 for cls in classes} for i in range(len(Y))]
        scores = [{cls:0 for cls in classes} for i in range(len(Y))]
        for i in range(no_classes):
            for j in range(i+1, no_classes):
                for k in range(len(Y)):
                    if all_preds[i][j][k] != -1:
                        frequencies[k][int(all_preds[i][j][k])] += 1
                        scores[k][int(all_preds[i][j][k])] += all_scores[i][j][k]
        
        def get_cls_max(freq_dict, score_dict):
            max_freq = 0
            max_score = -1
            cls_max = -1
            for cls in freq_dict.keys():
                if freq_dict[cls] > max_freq:
                    max_freq = freq_dict[cls]
                    max_score = score_dict[cls]
                    cls_max = cls
                elif freq_dict[cls] == max_freq and cls_max!=-1:
                    if score_dict[cls] > max_score:
                        max_freq = freq_dict[cls]
                        max_score = score_dict[cls]
                        cls_max = cls
            return cls_max
        
        voted_preds = [get_cls_max(frequencies[i], scores[i]) for i in range(len(frequencies))]
        no_correct = len([1 for i in range(len(voted_preds)) if voted_preds[i] == Y[i]])
        accuracy = no_correct / len(voted_preds)

        self.result_dict["accuracy"] = "{:.3f}".format(accuracy * 100)
        self.result_dict["time"] = "{:.3f}s".format(self.train_time)
        self.result_dict["predictions"] = [int(i) for i in voted_preds]
        self.result_dict["labels"] = [int(y) for y in Y]
        with open(result_save, "w") as f:
            json.dump(self.result_dict, f)

        return accuracy

    def train_libsvm(self, kernel="gaussian"):
        self.train_time = time.time()
        if kernel.lower() == "linear":
            self.libsvm_model = svmutil.svm_train(self.Y, self.X, '-c {} -t 0 -q'.format(self.C))
        elif kernel.lower() == "gaussian":
            self.libsvm_model = svmutil.svm_train(self.Y, self.X, '-c {} -t 2 -g 0.05 -q'.format(self.C))
        self.train_time = time.time() - self.train_time

    def test_libsvm(self, data_path=None, kernel="gaussian", result_save=None, model=None):
        if model is None:
            model = self.libsvm_model
        if data_path is None:
            X = self.X
            Y = self.Y
        else:
            X, Y = self.load_data(data_path, to_return=True)
        p_label, p_acc, p_val = svmutil.svm_predict(Y, X, model)

        if result_save is not None:
            self.result_dict["accuracy"] = "{:.3f}".format(p_acc[0])
            self.result_dict["time"] = "{:.3f}s".format(self.train_time)
            
            p_val = [[p for p in p1] for p1 in p_val]
            self.result_dict["predictions"] = p_val
            self.result_dict["labels"] = [int(y) for y in p_label]
            with open(result_save, "w") as f:
                json.dump(self.result_dict, f)
        return p_acc[0]

    def draw_confusion(self, result_json, save_path):
        with open(result_json) as f:
            result_dict = json.load(f)
        preds = result_dict["predictions"]
        labels = result_dict["labels"]

        classes = list(set(labels + preds))
        confusion_matrix = [[0 for cls1 in classes] for cls2 in classes]
        cls_mapping = {classes[i]: i for i in range(len(classes))}
        for i in range(len(preds)):
            confusion_matrix[cls_mapping[preds[i]]][cls_mapping[labels[i]]] += 1
        
        df_cm = pd.DataFrame(confusion_matrix, index = classes, columns = classes)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
        plt.ylabel("predictions")
        plt.xlabel("labels")
        plt.title("Confusion Matrix")
        plt.savefig(save_path)
        plt.close()

    def visualise_mistakes(self, result_json, save_dir, input_path):
        with open(result_json) as f:
            result_dict = json.load(f)
        preds = result_dict["predictions"]
        labels = result_dict["labels"]
        X, _ = self.load_data(input_path, to_return=True)
        wrong_indices = np.asarray([i for i in range(len(preds)) if preds[i] != labels[i]])
        wrong_x = np.asarray(X[wrong_indices, :])
        wrong_imgs = 255*wrong_x.reshape((wrong_x.shape[0], 28, 28))

        for i in range(len(wrong_imgs)):
            save_path = os.path.join(save_dir, "{}_{}_{}.png".format(i, labels[wrong_indices[i]], preds[wrong_indices[i]]))
            cv2.imwrite(save_path, wrong_imgs[i])

    def k_fold_cross_val(self, k=5):
        m = len(self.Y)
        length_fold = m//k
        indices = np.asarray([i for i in range(m)])
        np.random.seed(0)
        np.random.shuffle(indices)
        all_accuracies = []
        best_model = None
        best_accuracy = -1
        for i in range(k):
            if i<k-1:
                fold_indices = indices[i*length_fold:(i+1)*length_fold]
            else:
                fold_indices = indices[i*length_fold:]
            non_fold_indices = np.asarray([i for i in indices if i not in fold_indices])
            X_train, Y_train = self.X[non_fold_indices, :], self.Y[non_fold_indices]
            X_val, Y_val = self.X[fold_indices, :], self.Y[fold_indices]
            libsvm_model = svmutil.svm_train(Y_train, X_train, '-c {} -t 2 -g 0.05 -q'.format(self.C))
            _, acc, _ = svmutil.svm_predict(Y_val, X_val, libsvm_model)
            out_file = open("log.txt", "a")
            out_file.write(str(acc[0]) + "\n")
            out_file.close()
            all_accuracies.append(acc[0])
            if acc[0]>best_accuracy:
                best_accuracy = acc[0]
                best_model = libsvm_model

        print(all_accuracies)
        return sum(all_accuracies)/len(all_accuracies), best_model

    def plot_cross_val(self, result_json, save_path="cv_accuracies.png"):
        with open(result_json) as f:
            result_dict = json.load(f)
        cv_acc = result_dict["crossval_accuracies"]
        test_acc = result_dict["test_accuracies"]
        c_values = result_dict["C_values"]
        c_log_values = [np.log10(c) for c in c_values]
        plt.plot(c_log_values, cv_acc, label="Cross Validation Accuracy")
        plt.plot(c_log_values, test_acc, label="Test Accuracy")
        plt.legend()
        plt.title("Variation of Accuracies over different values of C")
        plt.xlabel("Parameter C (log scale)")
        plt.ylabel("Accuracy")
        plt.savefig(save_path, dpi=300)


def get_args():
    # for getting command line arguments
    parser = argparse.ArgumentParser(description='Gradient Descent')
    parser.add_argument('--train_path', type=str, default="./data/Music_Review_train.json", help='the file path where training data is stored')
    parser.add_argument('--test_path', type=str, default="./data/Music_Review_test.jso", help='the file path where test data is stored')
    parser.add_argument('--part', type=str, default="a", help='the part to be run')
    args = parser.parse_args()
    return args

def main():

    os.makedirs("Q2/results", exist_ok=True)
    os.makedirs("Q2/plots", exist_ok=True)
    os.makedirs("Q2/wrongs/cvxopt/", exist_ok=True)

    args = get_args()

    print("Running Part {} of question 2 multiclass".format(args.part.lower()))

    ## Part a
    if args.part.lower() == "a":
        model = SVMMultiClassClassifier()
        model.load_data(args.train_path)
        model.init_classifiers()
        model.set_classifiers_data()
        model.train()
        print(model.test(result_save = "Q2/results/result_train.json"))
        print(model.test(args.test_path, result_save = "Q2/results/result_test.json"))

    ### Part b
    elif args.part.lower() == "b":
        model = SVMMultiClassClassifier(C=1)
        model.load_data(args.train_path)
        model.train_libsvm()
        print(model.test_libsvm(result_save = "Q2/results/result_train_libsvm.json"))
        print(model.test_libsvm(args.test_path, result_save = "Q2/results/result_test_libsvm.json"))

    ### part c
    elif args.part.lower() == "c":
        model = SVMMultiClassClassifier()
        model.draw_confusion("Q2/results/result_test.json", "Q2/plots/confusion1.pdf")
        model.draw_confusion("Q2/results/result_binary.json", "Q2/plots/confusion2.pdf")
        model.visualise_mistakes("Q2/results/result_test.json", "Q2/wrongs/cvxopt/", args.test_path)

    ### part d
    elif args.part.lower() == "d":
        model = SVMMultiClassClassifier()
        model.load_data(args.train_path)
    
        c_values = [ 10**(-5), 10**(-3), 1, 5, 10]
        kfold_accs = []
        test_accs = []
        for c in c_values:
            model.set_c(c)
            k_acc, best_model = model.k_fold_cross_val()
            kfold_accs.append(k_acc)
            test_accs.append(model.test_libsvm(args.test_path, model=best_model))
        result_dict = {}
        result_dict["C_values"] = c_values
        result_dict["crossval_accuracies"] = kfold_accs
        result_dict["test_accuracies"] = test_accs
        with open("Q2/results/kfold_C_results.json", "w") as f:
            json.dump(result_dict, f)

        model = SVMMultiClassClassifier()
        model.plot_cross_val("Q2/results/kfold_C_results.json", save_path="Q2/plots/cv_accuracies.png")

    else:
        raise ValueError("The part should be one of a-d, got {}.".format(args.part))


if __name__ == "__main__":
    main()