import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt


class RandomForest():
    '''
       
        Parameters: 
            n_estimators        : number of trees in the ensemble
            max_features        : the number of features to consider when looking for the best split
            min_samples_split   : the minimum number of samples required to split an internal node
            random_state        : seed for random number generation. Used to make code reproducable
    '''
    def __init__(self, n_estimators, max_features, min_samples_split, random_state=0):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.X = None
        self.Y = None
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators = self.n_estimators, 
            max_features=self.max_features, 
            min_samples_split=self.min_samples_split, 
            criterion = 'gini', 
            random_state = self.random_state, 
            oob_score = True,
        )

    def __str__(self):
        return "Random forest classifier with {} estimators, {} max features and {} min samples split".format(self.n_estimators, self.max_features, self.min_samples_split)

    def encode_one_hot(self, X=None):
        if X is None:
            X = self.X
        new_x_list = []
        for i in range(X.shape[1]):
            attributes = X[:, i]
            if type(attributes[0]) == int:
                new_x_list.append(attributes)
            else:
                options = list(set(attributes))
                options.sort()
                for o in options:
                    new_x_list.append((attributes == o).astype("uint8"))
        
        X = np.asarray(new_x_list)
        X = np.transpose(X)
        return X

    def load_data(self, data_path, train=True):
        df = pd.read_csv(data_path, delimiter=";")
        X = df.iloc[:, :-1].values
        X = self.encode_one_hot(X)
        Y = df.iloc[:, -1].values
        if train:
            self.X = X
            self.Y = Y
        else:
            return X, Y

    def train(self, X=None, Y=None):
        if X is None or Y is None:
            assert self.X is not None and self.Y is not None
            X, Y = self.X, self.Y
        else:
            X, Y = X, Y
        self.model.fit(X, Y)

    def evaluate(self, mode="train", dataset_path=None, model=None):
        if mode == "train":
            X, Y = self.X, self.Y
        elif mode == "val" or mode == "test":
            X, Y = self.load_data(dataset_path, train=False)
        else:
            raise ValueError("Mode must be one of ['train', 'val', 'test', got {}".format(mode))
        if model is None:
            model = self.model
        preds = model.predict(X)
        accuracy = sum([preds[i] == Y[i] for i in range(len(preds))]) / len(Y)
        return accuracy

    def grid_search(self, n_estimators_values, max_features_values, min_samples_split_values, return_params=False):

        param_set = [(i, j, k) for k in min_samples_split_values for j in max_features_values for i in n_estimators_values]
        oob_max = -1
        best_classifier = None
        best_params = None
        for i in range(len(param_set)):
            print(i)
            n_estimators, max_features, min_samples_split = param_set[i]
            classifier = RandomForestClassifier(
                n_estimators = n_estimators, 
                max_features=max_features, 
                min_samples_split=min_samples_split, 
                criterion = 'gini', 
                random_state = self.random_state, 
                oob_score = True,
            )
            classifier.fit(self.X, self.Y)
            if classifier.oob_score_ > oob_max:
                oob_max = classifier.oob_score_
                best_classifier = classifier
                best_params = param_set[i]

        self.model = best_classifier
        self.n_estimators, self.max_features, self.min_samples_split = best_params
        if return_params:
            return best_classifier, best_params
        else:
            return best_classifier

    def plot_sensitivity(self, test_data_path, val_data_path, n_estimators_values, max_features_values, min_samples_split_values, save_format=None):
        n_estimator_results = []
        for n_estimator in n_estimators_values:
            classifier = RandomForestClassifier(
                n_estimators = n_estimator, 
                max_features=self.max_features, 
                min_samples_split=self.min_samples_split, 
                criterion = 'gini', 
                random_state = self.random_state, 
                oob_score = True,
            )
            classifier.fit(self.X, self.Y)
            n_estimator_results.append({
                "n_estimators": n_estimator,
                "test": 100*self.evaluate("test", test_data_path, model=classifier),
                "val": 100*self.evaluate("val", val_data_path, model=classifier),
            })

        max_features_results = []
        for max_features in max_features_values:
            classifier = RandomForestClassifier(
                n_estimators = self.n_estimators, 
                max_features=max_features, 
                min_samples_split=self.min_samples_split, 
                criterion = 'gini', 
                random_state = self.random_state, 
                oob_score = True,
            )
            classifier.fit(self.X, self.Y)
            max_features_results.append({
                "max_features": max_features,
                "test": 100*self.evaluate("test", test_data_path, model=classifier),
                "val": 100*self.evaluate("val", val_data_path, model=classifier),
            })

        min_samples_split_results = []
        for min_samples_split in min_samples_split_values:
            classifier = RandomForestClassifier(
                n_estimators = self.n_estimators, 
                max_features=self.max_features, 
                min_samples_split=min_samples_split, 
                criterion = 'gini', 
                random_state = self.random_state, 
                oob_score = True,
            )
            classifier.fit(self.X, self.Y)
            min_samples_split_results.append({
                "min_samples_split": min_samples_split,
                "test": 100*self.evaluate("test", test_data_path, model=classifier),
                "val": 100*self.evaluate("val", val_data_path, model=classifier),
            })

        if save_format is None:
            return 
        
        plt.plot([r["n_estimators"] for r in n_estimator_results], [r["test"] for r in n_estimator_results], label = "test")
        plt.plot([r["n_estimators"] for r in n_estimator_results], [r["val"] for r in n_estimator_results], label = "val")
        plt.scatter([r["n_estimators"] for r in n_estimator_results], [r["test"] for r in n_estimator_results])
        plt.scatter([r["n_estimators"] for r in n_estimator_results], [r["val"] for r in n_estimator_results])
        plt.legend()
        plt.xlabel("number of estimators")
        plt.ylabel("Accuracies")
        plt.ylim(89, 91)
        plt.savefig(save_format.format("n_estimators"), dpi=300)
        plt.close()

        plt.plot([r["max_features"] for r in max_features_results], [r["test"] for r in max_features_results], label = "test")
        plt.plot([r["max_features"] for r in max_features_results], [r["val"] for r in max_features_results], label = "val")
        plt.scatter([r["max_features"] for r in max_features_results], [r["test"] for r in max_features_results])
        plt.scatter([r["max_features"] for r in max_features_results], [r["val"] for r in max_features_results])
        plt.legend()
        plt.xlabel("Maximum Features")
        plt.ylabel("Accuracies")
        plt.ylim(89, 91)
        plt.savefig(save_format.format("max_features"), dpi=300)
        plt.close()

        plt.plot([r["min_samples_split"] for r in min_samples_split_results], [r["test"] for r in min_samples_split_results], label = "test")
        plt.plot([r["min_samples_split"] for r in min_samples_split_results], [r["val"] for r in min_samples_split_results], label = "val")
        plt.scatter([r["min_samples_split"] for r in min_samples_split_results], [r["test"] for r in min_samples_split_results])
        plt.scatter([r["min_samples_split"] for r in min_samples_split_results], [r["val"] for r in min_samples_split_results])
        plt.legend()
        plt.xlabel("Minimum Samples Split")
        plt.ylabel("Accuracies")
        plt.ylim(89, 91)
        plt.savefig(save_format.format("min_samples_split"), dpi=300)
        plt.close()
        

        

def main():
    os.makedirs("plots2", exist_ok=True)
    model = RandomForest(n_estimators=450, max_features=0.7, min_samples_split=2)
    model.load_data("./bank_dataset/bank_train.csv")
    
    
    # model.train()
    # print(model)
    # print("Train: {:.2f}%".format(100*model.evaluate()))
    # print("Out of Box accuracy: {:.2f}%".format(100*model.model.oob_score_))
    # print("Validation: {:.2f}%".format(100*model.evaluate("val", "./bank_dataset/bank_val.csv")))
    # print("Test: {:.2f}%".format(100*model.evaluate("test", "./bank_dataset/bank_test.csv")))
    
    
    
    # model.grid_search(
    #     n_estimators_values = np.arange(50, 450, 100),
    #     max_features_values = np.arange(0.1, 1, 0.2),
    #     min_samples_split_values = np.arange(2, 10, 2),
    # )
    # print("\n========================================")
    # print(model)
    # print("Train: {:.2f}%".format(100*model.evaluate()))
    # print("Validation: {:.2f}%".format(100*model.evaluate("val", "./bank_dataset/bank_val.csv")))
    # print("Test: {:.2f}%".format(100*model.evaluate("test", "./bank_dataset/bank_test.csv")))


    model.plot_sensitivity(
        "./bank_dataset/bank_test.csv",
        "./bank_dataset/bank_val.csv",
        n_estimators_values = np.arange(50, 450, 100),
        max_features_values = np.arange(0.1, 1, 0.2),
        min_samples_split_values = np.arange(2, 10, 2),
        save_format = "./plots2/sensitivity_{}.png"
    )

if __name__ == "__main__":
    main()
   