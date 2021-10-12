from collections import defaultdict, deque
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
import sys
import math
import time
import heapq

def freq_dict(l):
    d = defaultdict(int)
    for i in l:
        d[i] += 1
    return d

def most_frequent(d):
    return max(d.items(), key=itemgetter(1))[0]

def choose_attribute(x, is_bool):
    j_best, split_value, min_entropy = -1, -1, math.inf
    y = x[:, -1]
    mn, mx = np.amin(x, axis=0), np.amax(x, axis=0)
    for j in range(len(is_bool)):
        w, med = x[:, j], 0.5
        if not is_bool[j]: med = np.median(w)
        if mn[j] == mx[j] or med == mx[j]: continue
        y_split = [y[w <= med], y[w > med]]
        entropy, p = 0, 1 / len(y)
        for y_ in y_split:
            h, prob = 0, 1 / len(y_)
            counts = np.unique(y_, return_counts=True)[1].astype('float32') * prob
            entropy -= p * np.sum(counts * np.log(counts)) * len(y_)
        if entropy < min_entropy:
            min_entropy = entropy
            j_best = j
            split_value = med
    if j_best == -1:
        return -1, None, None, None, None, None
    left = x[x[:, j_best] <= split_value]
    right = x[x[:, j_best] > split_value]
    return j_best, split_value, left, right

class Node:
    """
    Node class for the decision tree
    Data:
    self.left, self.right: left and right children
    self.parent: parent of the node, None if this node is root
    self.is_leaf: boolean
    self.attribute_num: attribute number being split on - if self.is_leaf is False
    self.class_freq: class frequencies if self.is_leaf is True
    self.cl: class decision if self.is_leaf is True
    self.x: data associated to this node if self.is_leaf is True (used while growing tree)
    self.split_value: value to split on
    self.correct: correctly classified validation datapoints
    self.correct_ifleaf: correctly classified validation datapoints if it were a leaf
    """

    def __init__(self, x, x_test=None, x_valid=None, par=None):
        self.parent = par
        self.left = None
        self.right = None
        self.attribute_num = -1
        self.is_leaf = True
        self.x = x
        self.x_test = x_test
        self.x_valid = x_valid
        self.class_freq = freq_dict(x[:, -1])
        self.cl = most_frequent(self.class_freq)
        self.split_value = None
        self.correct = 0
        self.correct_ifleaf = 0
        self.correct_test = 0
        self.correct_ifleaf_test = 0
        self.correct_train = 0
        self.correct_ifleaf_train = 0
        self.is_deleted = False

    def __lt__(self, node):
        return node.correct < self.correct

class DecisionTree:
    """
    Decision tree class
    Data:
    self.root: root node of the tree
    self.train_accuracies: training accuracies found while training the model
    self.test_accuracies: test accuracies found while training the model
    self.valid_accuracies: validation accuracies found while training the model
    """
    # D is a numpy array, last col is y
    # is_bool: list of booleans: True if data is boolean, False if data is int
    # threshold: threshold for training accuracy
    def __init__(self,
                 D_train=None,
                 D_test=None,
                 D_valid=None,
                 is_bool=None,
                 threshold=1.0,
                 pruning=False,
                 max_nodes=math.inf):
        """
        Constructor for a DecisionTree
        Parameters:
        -----------------------------------------------------------------------
        D_train, D_test, D_valid: numpy arrays denoting train, test and val data
        is_bool: indicator for each column whether it is boolean or not
        threshold: accuracy till which the model needs to run
        pruning: boolean indicating whether pruning needs to be done or not
        max_nodes: maximum nodes allowed in the tree
        """
        self.train_accuracies = []
        self.test_accuracies = []
        self.valid_accuracies = []
        self.num_classes = int(D_train[:, -1].max()) + 1
        self.valid_accuracies_after_pruning = []
        self.train_accuracies_after_pruning = []
        self.test_accuracies_after_pruning = []
        self.pruned_tree_sizes = []

    def train(self):
        if self.S_train is not None:
            self.grow_tree()
        else:
            self.root = None


    def predict(self, D_test):
        """
        Predict labels of the given data using the model
        """
        predicted = []
        for x in D_test:
            node = self.root
            while not node.is_leaf:
                if x[node.attribute_num] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            predicted.append(node.cl)
        return np.array(predicted)

    def grow_tree(self,
                  D_train,
                  D_test,
                  D_valid,
                  is_bool,
                  threshold,
                  pruning,
                  max_nodes):
        """
        Create the tree
        Parameters:
        ------------------------------------------------------------------------
        D_train, D_test, D_valid: numpy arrays denoting train, test and val data
        is_bool: indicator for each column whether it is boolean or not
        threshold: accuracy till which the model needs to run
        pruning: boolean indicating whether pruning needs to be done or not
        max_nodes: maximum nodes allowed in the tree
        Raises:
        Exception 'Empty data' if D_train is empty
        """

        # empty data
        if len(D_train) == 0:
            raise Exception('Empty data')

        self.root = Node(x=D_train, x_test=D_test, x_valid=D_valid)
        q = deque()
        q.appendleft(self.root)
        node_list = []
        node_list.append(self.root)
        total_nodes = 1
        predictions_completed = 0
        train_accuracy, test_accuracy, valid_accuracy = 0, 0, 0
        y_train, y_test, y_valid = D_train[:, -1], D_test[:, -1], D_valid[:, -1]

        total_valid = D_valid.shape[0]
        total_test = D_test.shape[0]
        total_train = D_train.shape[0]

        def cnt_help(arr, element):
            return np.count_nonzero(arr == element)
        def cnt(n):
            return cnt_help(n.x[:, -1], n.cl)
        def cnt_t(n):
            return cnt_help(n.x_test[:, -1], n.cl)
        def cnt_v(n):
            return cnt_help(n.x_valid[:, -1], n.cl)

        total_correct_train = cnt(self.root)
        total_correct_test = cnt_t(self.root)
        total_correct_valid = cnt_v(self.root)

        while train_accuracy < threshold and q and total_nodes < max_nodes:

            node = q.pop()

            # if node is pure
            if len(node.class_freq) == 1:

                node.x = None

            else:

                j, node.split_value, left_x, right_x = choose_attribute(node.x, is_bool)

                if j == -1:
                    node.x = None
                    continue

                left_x_test = node.x_test[node.x_test[:, j] <= node.split_value]
                left_x_valid = node.x_valid[node.x_valid[:, j] <= node.split_value]
                right_x_test = node.x_test[node.x_test[:, j] > node.split_value]
                right_x_valid = node.x_valid[node.x_valid[:, j] > node.split_value]

                node.attribute_num = j
                node.is_leaf = False
                node.left = Node(x=left_x, x_test=left_x_test, x_valid=left_x_valid, par=node)
                node.right = Node(x=right_x, x_test=right_x_test, x_valid=right_x_valid, par=node)
                q.appendleft(node.left)
                q.appendleft(node.right)
                node_list.append(node.left)
                node_list.append(node.right)
                total_nodes += 2

                # find number of elements correct in left
                # find number of elements correct in right
                # find number of elements correct in current
                # add difference of (left + right) - cur

                train_diff = -cnt(node) + cnt(node.left) + cnt(node.right)
                test_diff = -cnt_t(node) + cnt_t(node.left) + cnt_t(node.right)
                valid_diff = -cnt_v(node) + cnt_v(node.left) + cnt_v(node.right)

                total_correct_train += train_diff
                total_correct_test += test_diff
                total_correct_valid += valid_diff
                train_accuracy = total_correct_train / total_train
                test_accuracy = total_correct_test / total_test
                valid_accuracy = total_correct_valid / total_valid
                self.train_accuracies.append(100 * train_accuracy)
                self.test_accuracies.append(100 * test_accuracy)
                self.valid_accuracies.append(100 * valid_accuracy)

                node.x, node.class_freq = None, None
                node.x_test, node.x_valid = None, None

        # finally discard all data in leaf nodes
        for node in node_list:
            node.x = None
            node.x_valid = None
            node.x_test = None

        if not pruning:
            return

        # now pass validation data through the node using dfs, and compute the confusion matrices at each node
        # compute the accuracy change at each non-leaf node
        # sort the nodes according to accuracy changes
        # remove nodes greedily as follows:
        # pop node from heap
        # if node is deleted or node's latest value is not the same as the other member of the pair, continue
        # if found a node that doesn't increase validation accuracy, stop
        # else remove node and all members of the subtree
        # also set the left and right children of this node to None
        # change correct, is_leaf of this node
        # then propagate to all ancestors of the node
        # then compute total accuracy using the root node

        # computes correctly classified at each node
        # option = 1, 2, 3 correspond to train, test and val respectively
        def compute_correct(n, data, option=3):
            computed_value = cnt_help(data[:, -1], n.cl)
            if option == 3:
                n.correct_ifleaf = computed_value
            elif option == 2:
                n.correct_ifleaf_test = computed_value
            else:
                n.correct_ifleaf_train = computed_value
            if not n.is_leaf:
                data_left = data[data[:, n.attribute_num] <= n.split_value]
                data_right = data[data[:, n.attribute_num] > n.split_value]
                computed_value = compute_correct(n.left, data_left, option) +\
                            compute_correct(n.right, data_right, option)
            if option == 3:
                n.correct = computed_value
            elif option == 2:
                n.correct_test = computed_value
            else:
                n.correct_train = computed_value
            return computed_value

        # recompute the confusion for each ancestor
        def propagate_confusion_upwards(n, heap):
            while n.parent is not None:
                n.parent.correct = n.parent.left.correct + n.parent.right.correct
                n.parent.correct_test = n.parent.left.correct_test + n.parent.right.correct_test
                n.parent.correct_train = n.parent.left.correct_train + n.parent.right.correct_train
                heapq.heappush(heap, (n.parent.correct - n.parent.correct_ifleaf, n.parent))
                n = n.parent

        compute_correct(self.root, D_valid, 3)
        compute_correct(self.root, D_test, 2)
        compute_correct(self.root, D_train, 1)

        # now create a heap, and put all nodes in it
        heap = []
        for node in node_list:
            if not node.is_leaf:
                heapq.heappush(heap, (node.correct - node.correct_ifleaf, node))

        def set_delete_subtree(n):
            n.is_deleted = True
            if n.is_leaf:
                return 1
            else:
                return 1 + set_delete_subtree(n.left) + set_delete_subtree(n.right)

        total = D_valid.shape[0]
        total_test = D_test.shape[0]
        total_train = D_train.shape[0]

        while heap:
            diff, n = heapq.heappop(heap)
            if n.is_deleted or (n.correct - n.correct_ifleaf != diff):
                continue
            if diff >= 0:
                break
            total_nodes -= set_delete_subtree(n)
            n.correct = n.correct_ifleaf
            n.correct_test = n.correct_ifleaf_test
            n.correct_train = n.correct_ifleaf_train
            n.is_leaf = True
            n.left = None
            n.right = None
            propagate_confusion_upwards(n, heap)
            self.valid_accuracies_after_pruning.append(100 * self.root.correct / total)
            self.train_accuracies_after_pruning.append(100 * self.root.correct_train / total_train)
            self.test_accuracies_after_pruning.append(100 * self.root.correct_test / total_test)
            self.pruned_tree_sizes.append(total_nodes)
        return


def mainA():

    train = np.loadtxt(sys.argv[1], delimiter=',', skiprows=2)
    test = np.loadtxt(sys.argv[2], delimiter=',', skiprows=2)
    valid = np.loadtxt(sys.argv[3], delimiter=',', skiprows=2)

    is_bool = [(False if i < 10 else True) for i in range(54)]

    decision_tree = DecisionTree(
            D_train=train,
            D_test=test,
            D_valid=valid,
            is_bool=is_bool,
            threshold=1.0,
            pruning=False)

    x = list(range(1, 2 * len(decision_tree.train_accuracies) + 1, 2))
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy (in %)')
    plt.plot(x, decision_tree.train_accuracies, label='Training accuracy')
    plt.plot(x, decision_tree.test_accuracies, label='Test accuracy')
    plt.plot(x, decision_tree.valid_accuracies, label='Validation accuracy')
    print('final train accuracy:', decision_tree.train_accuracies[-1])
    print('final test accuracy:', decision_tree.test_accuracies[-1])
    print('final validation accuracy:', decision_tree.valid_accuracies[-1])
    plt.legend()
    plt.savefig('decision_tree_accuracies.png')
    plt.close()

def mainB():

    train = np.loadtxt(sys.argv[1], delimiter=',', skiprows=2)
    test = np.loadtxt(sys.argv[2], delimiter=',', skiprows=2)
    valid = np.loadtxt(sys.argv[3], delimiter=',', skiprows=2)

    is_bool = [(False if i < 10 else True) for i in range(54)]

    decision_tree = DecisionTree(
            D_train=train,
            D_test=test,
            D_valid=valid,
            is_bool=is_bool,
            threshold=1.0,
            pruning=True)

    x = list(range(1, 2 * len(decision_tree.train_accuracies) + 1, 2))

    print('initial train accuracy:', decision_tree.train_accuracies[-1])
    print('initial test accuracy:', decision_tree.test_accuracies[-1])
    print('initial validation accuracy:', decision_tree.valid_accuracies[-1])

    print('post pruning train accuracy:', decision_tree.train_accuracies_after_pruning[-1])
    print('post pruning test accuracy:', decision_tree.test_accuracies_after_pruning[-1])
    print('post pruning validation accuracy:', decision_tree.valid_accuracies_after_pruning[-1])

    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy (in %)')
    plt.plot(x, decision_tree.train_accuracies, label='Training accuracy')
    plt.plot(x, decision_tree.test_accuracies, label='Test accuracy')
    plt.plot(x, decision_tree.valid_accuracies, label='Validation accuracy')
    plt.legend()
    plt.savefig('decision_tree_accuracies.png')
    plt.close()
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy (in %)')
    plt.plot(decision_tree.pruned_tree_sizes, decision_tree.valid_accuracies_after_pruning, label='Validation accuracy')
    plt.plot(decision_tree.pruned_tree_sizes, decision_tree.train_accuracies_after_pruning, label='Training accuracy')
    plt.plot(decision_tree.pruned_tree_sizes, decision_tree.test_accuracies_after_pruning, label='Test accuracy')
    plt.legend()
    plt.xlim(90000, 50000)
    plt.savefig('decision_tree_post_pruning.png')

def mainC():

    train = np.loadtxt(sys.argv[1], delimiter=',', skiprows=2)
    test = np.loadtxt(sys.argv[2], delimiter=',', skiprows=2)
    valid = np.loadtxt(sys.argv[3], delimiter=',', skiprows=2)

    from sklearn.ensemble import RandomForestClassifier

    scores = []

    possible_n_estimators = [50, 150, 250, 350, 450] # 50 to 450
    possible_max_features = [0.1, 0.3, 0.5, 0.7, 0.9] # 0.1 to 1.0
    possible_min_samples_split = [2, 4, 6, 8, 10] # 2 to 10

    best_oob_score = -1
    best_n_estimators, best_min_samples_split, best_max_features = -1, -1, -1
    best_model = None

    for n_estimators in possible_n_estimators:
        for max_features in possible_max_features:
            for min_samples_split in possible_min_samples_split:
                t = time.time()
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             max_features=max_features,
                                             min_samples_split=min_samples_split,
                                             bootstrap=True,
                                             oob_score=True,
                                             n_jobs=4)
                clf.fit(train[:, :-1], train[:, -1])
                oob_score = clf.oob_score_
                print(n_estimators, max_features, min_samples_split, ':', oob_score)
                if oob_score > best_oob_score:
                    best_oob_score = oob_score
                    best_n_estimators = n_estimators
                    best_max_features = max_features
                    best_min_samples_split = min_samples_split
                    best_model = clf

    print(best_n_estimators, best_max_features, best_min_samples_split)
    print('oob score:', best_oob_score)
    y_pred_test = best_model.predict(test[:, :-1])
    y_pred_valid = best_model.predict(valid[:, :-1])
    y_pred_train = best_model.predict(train[:, :-1])
    print('training:', (y_pred_train == train[:, -1]).sum() / len(train[:, -1]))
    print('validation:', (y_pred_valid == valid[:, -1]).sum() / len(valid[:, -1]))
    print('test', (y_pred_test == test[:, -1]).sum() / len(test[:, -1]))

def mainD():

    train = np.loadtxt(sys.argv[1], delimiter=',', skiprows=2)
    test = np.loadtxt(sys.argv[2], delimiter=',', skiprows=2)
    valid = np.loadtxt(sys.argv[3], delimiter=',', skiprows=2)

    from sklearn.ensemble import RandomForestClassifier

    scores = []

    possible_n_estimators = [50, 150, 250, 350, 450] # 50 to 450
    possible_max_features = [0.1, 0.3, 0.5, 0.7, 0.9] # 0.1 to 1.0
    possible_min_samples_split = [2, 4, 6, 8, 10] # 2 to 10

    def run_parameters(n_estimators, max_features, min_samples_split):
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_features=max_features,
                                     min_samples_split=min_samples_split,
                                     bootstrap=True,
                                     criterion='gini',
                                     oob_score=True,
                                     n_jobs=4)
        clf.fit(train[:, :-1], train[:, -1])
        oob_score = clf.oob_score_
        y_pred_test = clf.predict(test[:, :-1])
        y_pred_valid = clf.predict(valid[:, :-1])
        test_acc = (y_pred_test == test[:, -1]).sum() / len(test[:, -1])
        valid_acc = (y_pred_valid == valid[:, -1]).sum() / len(valid[:, -1])
        return ((n_estimators, max_features, min_samples_split), (oob_score, test_acc, valid_acc))

    answers = []
    answers.append(run_parameters(450, 0.7, 2))
    #print('answers:', answers)
    for n in [50, 150, 250, 350]:
        answers.append(run_parameters(n, 0.7, 2))
        #print('answers:', answers)
    for f in [0.1, 0.3, 0.5, 0.9]:
        answers.append(run_parameters(450, f, 2))
        #print('answers:', answers)
    for s in [4, 6, 8, 10]:
        answers.append(run_parameters(450, 0.7, s))
        #print('answers:', answers)

    x = dict()
    for (parameters, scores) in answers:
        x[parameters] = (100 * scores[0], 100 * scores[1], 100 * scores[2])

    n_test, n_val, n_oob = [], [], []
    f_test, f_val, f_oob = [], [], []
    s_test, s_val, s_oob = [], [], []

    for n in possible_n_estimators:
        oob, test_acc, val_acc = x[(n, 0.7, 2)]
        n_oob.append(oob)
        n_test.append(test_acc)
        n_val.append(val_acc)

    for f in possible_max_features:
        oob, test_acc, val_acc = x[(450, f, 2)]
        f_oob.append(oob)
        f_test.append(test_acc)
        f_val.append(val_acc)

    for s in possible_min_samples_split:
        oob, test_acc, val_acc = x[(450, 0.7, s)]
        s_oob.append(oob)
        s_test.append(test_acc)
        s_val.append(val_acc)

    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy (in %)')
    plt.plot(possible_n_estimators, n_oob, label='Out of bag')
    plt.plot(possible_n_estimators, n_test, label='Test')
    plt.plot(possible_n_estimators, n_val, label='Validation')
    plt.legend()
    plt.savefig('estimator_sensitivity.png')
    plt.close()
    plt.xlabel('Fraction of features used')
    plt.ylabel('Accuracy (in %)')
    plt.plot(possible_max_features, f_oob, label='Out of bag')
    plt.plot(possible_max_features, f_test, label='Test')
    plt.plot(possible_max_features, f_val, label='Validation')
    plt.legend()
    plt.savefig('feature_sensitivity.png')
    plt.close()
    plt.xlabel('Minimum samples needed for split')
    plt.ylabel('Accuracy (in %)')
    plt.plot(possible_min_samples_split, s_oob, label='Out of bag')
    plt.plot(possible_min_samples_split, s_test, label='Test')
    plt.plot(possible_min_samples_split, s_val, label='Validation')
    plt.legend()
    plt.savefig('min_samples_split_sensitivity.png')
    plt.close()

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def main():

    pruning = (sys.argv[1] == '2')

    train = np.loadtxt(sys.argv[2], delimiter=',', skiprows=2)
    test = np.loadtxt(sys.argv[4], delimiter=',', skiprows=1)
    valid = np.loadtxt(sys.argv[3], delimiter=',', skiprows=1)

    is_bool = [(False if i < 10 else True) for i in range(54)]

    decision_tree = DecisionTree(
            D_train=train,
            D_test=test,
            D_valid=valid,
            is_bool=is_bool,
            threshold=1.0,
            pruning=pruning)

    y_pred = decision_tree.predict(test[:, :-1])
    write_predictions(sys.argv[5], y_pred)

if __name__ == '__main__':
    main()