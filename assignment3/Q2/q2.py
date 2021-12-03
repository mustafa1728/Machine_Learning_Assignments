from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

class Activation():
    def __init__(self, act_type):
        self.act_type = act_type
    
    def __call__(self, x):
        if self.act_type == "relu":
            return np.maximum(x, 0)
        elif self.act_type =="sigmoid":
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation {}. Currently supported: ['relu', 'sigmoid']".format(self.act_type))

    def grad(self, x):
        x = self(x)
        if self.act_type == "relu":
            return (x > 0).astype("uint")
        elif self.act_type =="sigmoid":
            z = self(x)
            return z * (1 - z)
        else:
            raise ValueError("Unsupported activation {}. Currently supported: ['relu', 'sigmoid']".format(self.act_type))


class Criterion():
    def __init__(self, loss_type):
        self.loss_type = loss_type

    def mse(self, out, label):
        return np.mean((out - label)*(out - label)) / 2
    
    def __call__(self, out, label):
        if self.loss_type == "MSE":
            return self.mse(out, label)
        else:
            raise ValueError("Unsupported loss type {}. Currently supported: ['MSE']".format(self.loss_type))

class FullyConnectedLayer():
    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = Activation(activation)
        self.w =  np.random.randn(input_dim, output_dim)
        self.b = 0.2*np.random.rand(1, output_dim)
        # self.b = np.ones((1, output_dim))
    
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        self.input = input
        
        pre_activation = np.dot(input, self.w) + self.b
        post_activation = self.activation(pre_activation)
        return post_activation, pre_activation

    def backward(self, cur_delta_A, prev_A, cur_z, lr):
        cur_delta_z = cur_delta_A*self.activation.grad(cur_z)
        delta_W_curr = np.dot(np.transpose(prev_A), cur_delta_z)
        delta_b_curr = np.sum(cur_delta_z, axis=0, keepdims=True)
        self.w = self.w + lr*delta_W_curr#/cur_z.shape[0]
        self.b = self.b + lr*delta_b_curr#/cur_z.shape[0]
        delta_A = np.dot(cur_delta_z, self.w.T)
        return delta_A

        

class NeuralNetwork():
    def __init__(self, hidden_layers, feature_dim, n_classes, batch_size, activation="sigmoid"):
        self.hidden_layers = hidden_layers
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.layers = []
        self.layers.append(FullyConnectedLayer(feature_dim, hidden_layers[0], activation))
        for i in range(len(hidden_layers) - 1):
            self.layers.append(FullyConnectedLayer(hidden_layers[i], hidden_layers[i+1], activation))
        self.layers.append(FullyConnectedLayer(hidden_layers[-1], n_classes, "sigmoid"))

        self.layer_outputs = [None for i in self.layers]
        self.layer_pre_acts = [None for i in self.layers]

    def forward(self, x):
        for i in range(len(self.layers)):
            x, z = self.layers[i](x)
            self.layer_outputs[i] = copy.deepcopy(x)
            self.layer_pre_acts[i] = copy.deepcopy(z)
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, out, label, lr):
        delta = label - out
        delta = ((1/self.batch_size)*delta*self.layers[-1].activation.grad(self.layer_outputs[-1])) 
        for i in range(0, len(self.layers) - 1):
            l_no = len(self.layers) - 2 - i
            delta = self.layers[l_no+1].backward(delta, self.layer_outputs[l_no], self.layer_pre_acts[l_no+1], lr)


class Trainer():
    def __init__(self, model, max_iter, lr, lr_adaptive, batch_size, loss="MSE"):
        self.model = model
        self.max_iter = max_iter
        self.base_lr = lr
        self.lr = self.base_lr
        self.lr_adaptive = lr_adaptive
        self.batch_size = batch_size
        self.criterion = Criterion(loss)

    def encode_one_hot(self, X=None):
        if X is None:
            X = self.X
        new_x_list = []
        for i in range(X.shape[1]):
            attributes = X[:, i]
            options = list(set(attributes))
            options.sort()
            for o in options:
                new_x_list.append((attributes == o).astype("uint8"))
        
        X = np.asarray(new_x_list)
        X = np.transpose(X)
        return X

    def load_data(self, data_path, train=True, y_one_hot=True):
        df = pd.read_csv(data_path, delimiter=",", header=None)
        X = df.iloc[:, :-1].values
        X = self.encode_one_hot(X)
        Y = df.iloc[:, -1].values
        if y_one_hot:
            Y = self.encode_one_hot(Y.reshape(-1, 1))
        if train:
            self.X = X
            self.Y = Y
        else:
            return X, Y

    def save_transformed_dataset(self, save_path):
        data = np.concatenate((self.X, self.Y), axis = -1)
        df = pd.DataFrame(data=data)
        df.to_csv(save_path, index=False, header=None)

    def shuffle(self):
        # randomly shuffles the data
        idx = np.arange(self.Y.shape[0])
        np.random.shuffle(idx)
        self.X = self.X[idx, :]
        self.Y = self.Y[idx]

    def stopping(self, loss_history, last_k = 200):
        if len(loss_history) < 2*last_k:
            return False
        last_k_losses = loss_history[:-last_k]
        last_2k_losses = loss_history[-2*last_k:-last_k]
        avg_loss_1 = sum(last_k_losses)/(10**(-6)+len(last_k_losses))
        avg_loss_2 = sum(last_2k_losses)/(10**(-6)+len(last_2k_losses))
        if abs(avg_loss_2 - avg_loss_1) < 10**(-4):
            return True
        return False
        


    def train(self, verbose=0):
        start_time = time.time()
        loss_history = []
        
        self.no_batches = self.X.shape[0] // self.batch_size
        self.no_steps = 0
        self.no_epochs = 0
        while self.no_epochs < self.max_iter:
            self.no_epochs += 1
            self.shuffle()
            for batch in range(self.no_batches):
                self.no_steps += 1
                x_batch = self.X[batch * self.batch_size : (batch + 1) * self.batch_size]
                y_batch = self.Y[batch * self.batch_size : (batch + 1) * self.batch_size]
                out = self.model(x_batch)
                loss = self.criterion(out, y_batch)
                self.model.backward(out, y_batch, lr=self.lr)
                if self.stopping(loss_history) or self.no_epochs > self.max_iter:
                    break

            if self.lr_adaptive:
                self.lr = self.base_lr / (np.sqrt(1 + self.no_epochs))
            if (self.no_epochs+1)%1000 == 0:
                print("Accuracy: ", self.evaluate("./data/poker-hand-training-true.data"))
            if self.stopping(loss_history):
                break

            if verbose == 2 :
                print("Epoch {} | Loss {}".format(self.no_epochs+1, loss))
            loss_history.append(loss)
        if verbose:
            print("Total {} iterations".format(self.no_epochs+1))
            print("Final loss is {}".format(loss))
        self.training_time = time.time() - start_time


    def evaluate(self, dataset_path=None, model=None, return_reds=False):
        X, Y = self.load_data(dataset_path, train=False, y_one_hot=False)
        if model is None:
            model = self.model
        out = model(X)
        # print(out[:100][::5])
        preds = np.argmax(out, axis = -1)
        preds[Y == 1] = 1
        accuracy = sum([preds[i] == Y[i] for i in range(len(preds))]) / len(Y)
        if not return_reds:
            return accuracy
        else:
            return preds, Y

    def plot_confusion(self, dataset_path, save_path):
        preds, labels = self.evaluate(dataset_path, return_reds=True)
        classes = list(set(list(labels) + list(preds)))
        confusion_matrix = [[0 for cls1 in classes] for cls2 in classes]
        cls_mapping = {classes[i]: i for i in range(len(classes))}
        for i in range(len(preds)):
            confusion_matrix[cls_mapping[preds[i]]][cls_mapping[labels[i]]] += 1
        
        df_cm = pd.DataFrame(confusion_matrix, index = classes, columns = classes)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
        plt.ylabel("predictions")
        plt.xlabel("labels")
        plt.title("Confusion Matrix")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def train_eval_lib(self, train_dataset_path, test_dataset_path):
        X, Y = self.load_data(train_dataset_path, train=False, y_one_hot=False)
        classifier = MLPClassifier(
            random_state=1, 
            max_iter=500,
            solver = "sgd",
            activation = "logistic",
            batch_size=100, 
            learning_rate_init=0.1
        )
        classifier = classifier.fit(X, Y)
        train_accuracy = classifier.score(X, Y)
        X, Y = self.load_data(test_dataset_path, train=False, y_one_hot=False)
        test_accuracy = classifier.score(X, Y)
        return train_accuracy, test_accuracy







def main():
    # adaptive_lr = False
    # os.makedirs("./plots", exist_ok = True)
    # for adaptive_lr in [True, False]:
    #     for hidden_layer_size in [5, 10, 15, 20, 25]:
    #         model = NeuralNetwork(
    #             hidden_layers = [hidden_layer_size], 
    #             feature_dim = 85, 
    #             n_classes = 10, 
    #             batch_size = 100,
    #             activation="sigmoid",
    #         )
    #         trainer = Trainer(
    #             model,
    #             max_iter=200000,#2000 - 100*hidden_layer_size,
    #             lr=0.1,
    #             lr_adaptive=adaptive_lr,
    #             batch_size = 100,
    #             loss = "MSE",
    #         )
    #         trainer.load_data("./data/poker-hand-testing.data")
    #         trainer.save_transformed_dataset("./data/test_transformed.csv") 
    #         trainer.load_data("./data/poker-hand-training-true.data")
    #         trainer.save_transformed_dataset("./data/train_transformed.csv") 
    #         print("NN with hidden units {}".format(hidden_layer_size))
    #         trainer.train(verbose=1)
    #         print("Training accuracy {:.2f}".format(100*trainer.evaluate("./data/poker-hand-training-true.data")))
    #         print("Test accuracy {:.2f}".format(100*trainer.evaluate("./data/poker-hand-testing.data")))
    #         print("Training Time {}s".format(trainer.training_time))
            # if adaptive_lr:
            #     cnfs_path = "./plots/confusion_adaptive_units{}.png".format(hidden_layer_size)
            # else:
            #     cnfs_path = "./plots/confusion_units{}.png".format(hidden_layer_size)
            # trainer.plot_confusion("./data/poker-hand-testing.data", cnfs_path)

    # for activation in ["sigmoid", "relu"]:
    #     model = NeuralNetwork(
    #                 hidden_layers = [100, 100], 
    #                 feature_dim = 85, 
    #                 n_classes = 10, 
    #                 batch_size = 100,
    #                 activation=activation,
    #             )
    #     trainer = Trainer(
    #         model,
    #         max_iter=200000,
    #         lr=0.1,
    #         lr_adaptive=True,
    #         batch_size = 100,
    #         loss = "MSE",
    #     )
    #     trainer.load_data("./data/poker-hand-training-true.data")
    #     print("NN with activation {}".format(activation))
    #     trainer.train(verbose=1)
    #     print("Training accuracy {:.2f}".format(100*trainer.evaluate("./data/poker-hand-training-true.data")))
    #     print("Test accuracy {:.2f}".format(100*trainer.evaluate("./data/poker-hand-testing.data")))
    #     print("Training Time {}s".format(trainer.training_time))
        # cnfs_pa th = "./plots/confusion_{}.png".format(activation)
        # trainer.plot_confusion("./data/poker-hand-testing.data", cnfs_path)

    trainer = Trainer(
            None,
            max_iter=200000,
            lr=0.1,
            lr_adaptive=True,
            batch_size = 100,
            loss = "MSE",
        )
    lib_train, lib_test = trainer.train_eval_lib("./data/poker-hand-training-true.data", "./data/poker-hand-testing.data")
    print("The sklearn model has training accuracy {:.2f} and testing accuracy {:.2f}".format(100*lib_train, 100*lib_test))

    # no_units = [5, 10, 15, 20, 25]
    # train_accuracies = [89.70, 80.99, 79.42, 76.03, 76.80]
    # test_accuracies = [89.86, 80.92, 79.12, 76.01, 76.62]
    # times = [38.34, 29.15, 36.59, 34.43, 31.42]
    
    # plt.plot(no_units, train_accuracies)
    # plt.xlabel("Number of hidden layer units")
    # plt.ylabel("Training Accuracy")
    # plt.title("Training Accuracies for varying hidden units")
    # plt.savefig("train.png", dpi=300)
    # plt.close()

    # plt.plot(no_units, test_accuracies)
    # plt.xlabel("Number of hidden layer units")
    # plt.ylabel("Test Accuracy")
    # plt.title("Test Accuracies for varying hidden units")
    # plt.savefig("test.png", dpi=300)
    # plt.close()

    # plt.plot(no_units, times)
    # plt.xlabel("Number of hidden layer units")
    # plt.ylabel("Time to train")
    # plt.title("Time to train for varying hidden units")
    # plt.savefig("time.png", dpi=300)
    # plt.close()

    # no_units = [5, 10, 15, 20, 25]
    # train_accuracies = [92.33, 92.33, 92.33, 91.96, 92.20]
    # test_accuracies = [92.37, 92.37, 92.37, 92.03, 92.24]
    # times = [30.67, 26.89, 28.89, 29.62, 30.38]
    
    # plt.plot(no_units, train_accuracies)
    # plt.xlabel("Number of hidden layer units")
    # plt.ylabel("Training Accuracy")
    # plt.title("Training Accuracies for varying hidden units")
    # plt.savefig("train_adaptive.png", dpi=300)
    # plt.close()

    # plt.plot(no_units, test_accuracies)
    # plt.xlabel("Number of hidden layer units")
    # plt.ylabel("Test Accuracy")
    # plt.title("Test Accuracies for varying hidden units")
    # plt.savefig("test_adaptive.png", dpi=300)
    # plt.close()

    # plt.plot(no_units, times)
    # plt.xlabel("Number of hidden layer units")
    # plt.ylabel("Time to train")
    # plt.title("Time to train for varying hidden units")
    # plt.savefig("time_adaptive.png", dpi=300)
    # plt.close()


if __name__ == "__main__":
    main()
   