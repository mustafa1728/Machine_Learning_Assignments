import pandas as pd
import numpy as np

class TreeNode():
    def __init__(self, S = None, left=None, right=None, boolean_attributes=None, pred=None, condition=None):
        self.S = S
        self.left = left
        self.right = right
        self.boolean_attributes = boolean_attributes
        # Only leaf nodes should have a pred
        assert pred is None or (left is None and right is None)
        self.pred = pred
        self.condition = condition

    def is_leaf(self):
        return self.left is None and self.right is None

    @staticmethod
    def evaluate_split(S_left, S_right):
        return 100

    @staticmethod
    def split_numerical(values, x, y, best_attr):
        values = [x_i[best_attr] for x_i in x]
        median = np.median(np.asarray(values))
        def split_condition(x_list):
            return x_list[best_attr]>median
        indices_left = [i for i in range(len(x)) if split_condition(x[i])]
        indices_right = [i for i in range(len(x)) if not split_condition(x[i])]
        S_left = (x[indices_left], y[indices_left])
        S_right = (x[indices_right], y[indices_right])
        return S_left, S_right, split_condition

    @staticmethod
    def split_categorical(x, y, best_attr):
        values = [x_i[best_attr] for x_i in x]
        options = list(set(values))
        median = np.median(np.asarray(values))
        def split_condition(x_list):
            return x_list[best_attr]>median
        indices_left = [i for i in range(len(x)) if split_condition(x[i])]
        indices_right = [i for i in range(len(x)) if not split_condition(x[i])]
        S_left = (x[indices_left], y[indices_left])
        S_right = (x[indices_right], y[indices_right])
        return S_left, S_right, split_condition

    @staticmethod
    def grow_tree(S, boolean_attributes):
        print("here")
        x, y = S
        no_ones = sum([y_i==1 for y_i in y])
        if no_ones == 0:
            return TreeNode(S, pred=0)
        elif no_ones == len(y):
            return TreeNode(S, pred=1)
        else:
            best_val = -np.inf
            best_split = (None, None)
            for i in range(len(x[0])):
                if boolean_attributes[i] == 0:
                    S_left, S_right, split_condition = TreeNode.split_numerical(x, y, i)       
                else:
                    S_left, S_right, split_condition = TreeNode.split_categorical(x, y, i)
                val = TreeNode.evaluate_split(S_left, S_right)
                if val > best_val:
                    best_val=val
                    best_split = (S_left, S_right)
            (S_left, S_right) = best_split
            left_child = TreeNode.grow_tree(S_left, boolean_attributes)
            right_child = TreeNode.grow_tree(S_right, boolean_attributes)
            return TreeNode(S, left_child, right_child, boolean_attributes=boolean_attributes, condition=split_condition)


class DecisionTree:
    def __init__(self, boolean_attributes=None):
        self.S_train = None
        self.S_val = None
        self.S_test = None
        self.root = None
        self.boolean_attributes=boolean_attributes

    def load_data(self, data_path, type="train"):
        df = pd.read_csv(data_path, delimiter=";")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        y = np.asarray([1 if y[i]=="yes" else 0 for i in range(y.shape[0])])
        if self.boolean_attributes is None:
            self.boolean_attributes = [True for i in range(len(X[0]))]
        if type == "train":
            self.S_train = (X, y)
        elif type == "val":
            self.S_val = (X, y)
        elif type == "test":
            self.S_test = (X, y)
        else:
            raise ValueError("Type must be one of [train, val, test], got {}.".format(type))
    
    def train(self):
        self.root = TreeNode.grow_tree(self.S_train, self.boolean_attributes)

    def predict_single(self, x):
        node = self.root
        while not node.is_leaf():
            if node.condition(x):
                node = node.left
            else:
                node = node.right
        return node.pred






def main():
    model = DecisionTree()
    model.load_data("bank_dataset/bank_train.csv", "train")
    model.load_data("bank_dataset/bank_val.csv", "val")
    model.load_data("bank_dataset/bank_test.csv", "test")
    model.train()

if __name__ == "__main__":
    main()