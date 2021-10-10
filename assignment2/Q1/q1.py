import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import json
import time
import os

from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))
from nltk.stem.porter import PorterStemmer


class RandomClassifier():
    '''
        Random classifier class that predicts 
        a label randomly for each sample
    '''
    def __init__(self):
        self.result_dict = {}
        pass 

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            json_data = [json.loads(l) for l in f]

        ratings = [int(d["overall"]) for d in json_data]
        reviewTexts = [d["reviewText"] for d in json_data]
        self.classes = list(set(ratings))
        return reviewTexts, ratings

    def test(self, data_path, save_result="Q1/results/result_random.json"):
        x, y = self.load_data(data_path)
        preds = np.random.choice(self.classes, len(x))
        accuracy = len([i for i in range(len(preds)) if preds[i] == y[i]])/len(y)
        if save_result is not None:
            self.result_dict["time"] = 0
            self.result_dict["accuracy"] = 100*(accuracy)
            self.result_dict["labels"] = y
            self.result_dict["predictions"] = preds.tolist()
            with open(save_result, "w") as f:
                json.dump(self.result_dict, f)
        return accuracy

class MajorityClassifier():
    '''
        Majority classifier class that calculates
        the most frequent label in the train data 
        and always predicts that label for each sample
    '''
    def __init__(self):
        self.result_dict = {}
        pass 

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            json_data = [json.loads(l) for l in f]

        ratings = [int(d["overall"]) for d in json_data]
        reviewTexts = [d["reviewText"] for d in json_data]
        self.classes = list(set(ratings))
        frequencies = {cls: 0 for cls in self.classes}
        for r in ratings:
            frequencies[r] += 1
        self.majority_class = -1
        maj_freq = -1
        for cls in self.classes:
            if frequencies[cls] > maj_freq:
                self.majority_class = cls
                maj_freq = frequencies[cls]
        return reviewTexts, ratings

    def test(self, data_path, save_result="Q1/results/result_majority.json"):
        x, y = self.load_data(data_path)
        preds = [self.majority_class for i in x]
        accuracy = len([i for i in range(len(preds)) if preds[i] == y[i]])/len(y)
        if save_result is not None:
            self.result_dict["time"] = 0
            self.result_dict["accuracy"] = 100*(accuracy)
            self.result_dict["labels"] = y
            self.result_dict["predictions"] = preds
            with open(save_result, "w") as f:
                json.dump(self.result_dict, f)
        return accuracy

class NaiveBayesClassifier:
    '''
        Main class for this question
        Implements the Naive Bayes algorithm for textclassification
        Has option sw_stm for chosing whether or not to perform stemming and stop word removal
        Has option freq_thresh_type = None, hard and soft for new feature
        Can work with feature_type = bag_of_word and bigram
        Provides simple API for drawing confusion matrices and calculating class wise F1 scores.
    '''
    def __init__(self, alpha = 1, sw_stm = False):
        self.alpha = alpha
        self.X, self.y=Y = None, None
        self.vocab = []
        self.vocab_size = 0
        self.word_to_no = {}
        self.phi = {}
        self.theta = {}
        self.sw_stm = sw_stm
        self.result_dict = {}
        self.train_time = 0
        self.bi_grams = None

    def load_data(self, data_path, to_return=False, include_summaries=False):
        with open(data_path, 'r') as f:
            json_data = [json.loads(l) for l in f]

        reviewTexts = [self._process_single(d["reviewText"]) for d in json_data]
        ratings = [int(d["overall"]) for d in json_data]

        if self.sw_stm:
            # comment out for no stop words removal
            reviewTexts = [[word for word in x if not word in stop_words] for x in reviewTexts]
            stemmer = PorterStemmer()
            reviewTexts = [[stemmer.stem(word) for word in x] for x in reviewTexts]

        
        if include_summaries:
            summaries = [self._process_single(d["summary"]) for d in json_data]
            if self.sw_stm:
                stemmer = PorterStemmer()
                summaries = [[stemmer.stem(word) for word in x] for x in summaries]
            

        if to_return:
            return reviewTexts, ratings
        else:
            if include_summaries:
                self.X = [reviewTexts[i] + summaries[i] for i in range(len(summaries))]
            else:
                self.X = reviewTexts
            self.Y = ratings

    
    def _process_single(self, x):
        char_list = [ch.lower() for ch in x if ch.isalpha() or ch == " "]
        text_string = "".join(char_list)
        return [x for x in text_string.split(" ") if x!=""]

    def train(self, freq_thresh_type=None, freq_threshold = 5):
        self.train_time = time.time()
        word_list = [word for x in self.X for word in x]
        self.vocab = list(set(word_list))  
        if freq_thresh_type == "hard":
            word_freqs = {word: 0 for word in self.vocab}
            for word in word_list:
                word_freqs[word] += 1
            self.vocab = [word for word in word_freqs.keys() if word_freqs[word]>freq_threshold]
        elif freq_thresh_type == "soft":
            word_freqs = {word: 0 for word in self.vocab}
            for word in word_list:
                word_freqs[word] += 1
            self.vocab = [word for word in word_freqs.keys() if word_freqs[word]>freq_threshold]
            self.vocab_weights = {word: word_freqs[word]/freq_threshold for word in self.vocab}
        
        self.vocab_size = len(self.vocab)  
        self.word_to_no = {self.vocab[i]: i for i in range(len(self.vocab))}
        
        self.phi = {y: 0 for y in list(set(self.Y))}
        N = len(self.Y)
        for y in self.Y:
            self.phi[y] += 1/N
        
        
        class_wise_counts = {y: {word: 0 for word in self.vocab} for y in list(set(self.Y))}
        for x, y in zip(self.X, self.Y):
            for word in x:
                try:
                    if freq_thresh_type != "soft":
                        class_wise_counts[y][word] += 1
                    else:
                        class_wise_counts[y][word] += self.vocab_weights[word]
                except KeyError:
                    continue

        self.theta = {y: {word: 0 for word in self.vocab} for y in self.phi.keys()}
        for y in self.phi.keys():
            self.theta[y] = {word: class_wise_counts[y][word] + self.alpha for word in self.vocab}
            total = sum([class_wise_counts[y][word] for word in self.vocab])
            self.theta[y] = {word: self.theta[y][word] / total for word in self.vocab}
        
        self.train_time = time.time() - self.train_time

    def predict_single(self, input_string, to_process = True, ret_probs = False):
        if to_process:
            x = self._process_single(input_string)
        else:
            x = input_string
        classes = list(self.phi.keys())
        log_probs = {i: 0 for i in classes}

        cleaned_x = []
        for word in x:
            try:
                self.word_to_no[word]
            except (KeyError):
                continue
            cleaned_x.append(word)
        for cls in classes:
            log_probs[cls] = np.log(self.phi[cls])
            log_probs[cls] += sum([np.log(self.theta[cls][word]) for word in cleaned_x])
        max_log_prob = log_probs[classes[0]]
        for cls in classes:
            if log_probs[cls] >= max_log_prob:
                max_log_prob = log_probs[cls]
                class_max = cls
        if ret_probs:
            return class_max, log_probs
        else:
            return class_max

    def get_bigrams(self):
        bigrams = [[word[i] + " " + word[i+1] for i in range(len(word) - 1)] for word in self.X]
        self.X = [self.X[i] + bigrams[i] for i in range(len(self.X))]

    def predict_single_bigram(self, input_string, to_process = True, ret_probs = False):
        if to_process:
            x = self._process_single(input_string)
        else:
            x = input_string
        bi_gram_x = [x[i] + " " + x[i+1] for i in range(len(x) - 1)]
        return self.predict_single(x + bi_gram_x, to_process=False, ret_probs=ret_probs)
        

    def test(self, dataset_path=None, ret_cfm=False, save_result = None, feature_type="bag_of_word"):
        if dataset_path is None:
            X, Y = self.X, self.Y
        else:
            X, Y = self.load_data(dataset_path, to_return=True)
        total_correct = 0
        total = 0
        if feature_type == "bag_of_word":
            pred_fn = self.predict_single
        elif feature_type == "bigram":
            pred_fn = self.predict_single_bigram
        else:
            raise ValueError("Feature type should be one of ['bag_of_word' or 'word2vec', received {}".format(feature_type))
        predictions = [pred_fn(x, to_process=False) for x in X]
        for i in range(len(Y)):
            prediction = predictions[i]
            if prediction == Y[i]:
                total_correct += 1
            total += 1
        if save_result is not None:
            self.result_dict["time"] = self.train_time
            self.result_dict["accuracy"] = 100*(total_correct/total)
            self.result_dict["labels"] = Y
            self.result_dict["predictions"] = predictions
            with open(save_result, "w") as f:
                json.dump(self.result_dict, f)
        if ret_cfm:
            return (total_correct/total), predictions, Y
        else:
            return (total_correct/total)

    def confusion(self, result_dict_path, save_path=None, save_norm_path=None):
        with open(result_dict_path, "r") as f:
            result_dict = json.load(f)
        preds = result_dict["predictions"]
        labels = result_dict["labels"]
        classes = list(set(labels + preds))

        confusion_matrix = [[0 for cls1 in classes] for cls2 in classes]
        cls_mapping = {classes[i]: i for i in range(len(classes))}
        for i in range(len(preds)):
            confusion_matrix[cls_mapping[preds[i]]][cls_mapping[labels[i]]] += 1
        if save_path is None:
            return confusion_matrix
        df_cm = pd.DataFrame(confusion_matrix, index = classes, columns = classes)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
        plt.ylabel("predictions")
        plt.xlabel("labels")
        plt.title("Confusion Matrix")
        plt.savefig(save_path)
        plt.close()

        if save_norm_path is not None:
            cf_np = np.asarray(confusion_matrix)
            cf_np = 100*cf_np / np.sum(cf_np, axis = 0)
            cf_np = cf_np.astype("uint8")
            df_cm = pd.DataFrame(cf_np, index = classes, columns = classes)
            hm = sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
            for t in hm.texts: t.set_text(t.get_text() + " %")
            plt.ylabel("predictions")
            plt.xlabel("labels")
            plt.title("Confusion Matrix")
            plt.savefig(save_norm_path)
            plt.close()

    def compute_f1_class_wise(self, result_dict_path, save_path=None):
        with open(result_dict_path, "r") as f:
            result_dict = json.load(f)
        preds = result_dict["predictions"]
        labels = result_dict["labels"]

        classes = list(set(labels + preds))

        precisions = {cls: 0 for cls in classes}
        recalls = {cls: 0 for cls in classes}
        f1_scores = {cls: 0 for cls in classes}

        for cls in classes:
            no_true_positives = len([1 for i in range(len(labels)) if labels[i] == cls and preds[i] == cls])
            no_false_positives = len([1 for i in range(len(labels)) if labels[i] != cls and preds[i] == cls])
            no_false_negatives = len([1 for i in range(len(labels)) if labels[i] == cls and preds[i] != cls])

            precisions[cls] = no_true_positives/(10**(-10) + no_true_positives + no_false_positives)
            recalls[cls] = no_true_positives/(10**(-10) + no_true_positives + no_false_negatives)

            f1_scores[cls] = 2*precisions[cls]*recalls[cls]/(precisions[cls] + recalls[cls] + 10**(-10))
        f1_scores["macro"] = sum([f1_scores[cls] for cls in classes])/len([f1_scores[cls] for cls in classes])

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(f1_scores, f)
        return f1_scores

def get_args():
    # for getting command line arguments
    parser = argparse.ArgumentParser(description='Gradient Descent')
    parser.add_argument('--train_path', type=str, default="./data/Music_Review_train.json", help='the file path where training data is stored')
    parser.add_argument('--test_path', type=str, default="./data/Music_Review_test.jso", help='the file path where test data is stored')
    parser.add_argument('--part', type=str, default="a", help='the part to be run')
    args = parser.parse_args()
    return args

def main():

    os.makedirs("Q1/results", exist_ok=True)
    os.makedirs("Q1/plots", exist_ok=True)

    args = get_args()

    print("Running Part {} of question 1".format(args.part.lower()))

    ### Part a
    if args.part.lower() == "a":
        model = NaiveBayesClassifier(sw_stm=False, alpha=1)
        model.load_data(args.train_path)
        model.train()
        print("Training Complete!")
        print("length of vocabulary: ", model.vocab_size)
        train_acc = model.test(save_result="Q1/results/result_vanilla_train.json")
        print("Accuracy on training dataset: {:.4f}%".format(100*train_acc))
        test_acc = model.test(args.test_path, save_result="Q1/results/result_vanilla_test.json")
        print("Accuracy on test dataset: {:.4f}%".format(100*test_acc))

    ### Part b
    elif args.part.lower() == "b":
        model = RandomClassifier()
        model.load_data(args.train_path)
        print(model.test(args.train_path))
        print(model.test(args.test_path))

        model = MajorityClassifier()
        model.load_data(args.train_path)
        print(model.test(args.train_path))
        print(model.test(args.test_path))

    ### part c
    elif args.part.lower() == "c":
        model = NaiveBayesClassifier(sw_stm=True)
        model.confusion("Q1/results/result_vanilla_test.json", "Q1/plots/confusion.pdf", "Q1/plots/confusion_norm.pdf")

    ### part d
    elif args.part.lower() == "d":
        model = NaiveBayesClassifier(sw_stm=True)
        model.load_data(args.train_path)
        model.train()
        print("Training Complete!")
        print("length of vocabulary: ", model.vocab_size)
        test_acc = model.test(args.test_path, save_result="Q1/results/result_stm.json")
        print("Accuracy on test dataset: {:.4f}%".format(100*test_acc))
        model.confusion("Q1/results/result_stm.json", "Q1/plots/confusion_stm.pdf", "Q1/plots/confusion_norm_stm.pdf")

    ### part e
    elif args.part.lower() == "e":
        model = NaiveBayesClassifier(alpha=0.1, sw_stm=True)
        model.load_data(args.train_path)
        model.get_bigrams()
        model.train(freq_thresh_type="hard")
        print("Training Complete!")
        print("length of vocabulary: ", model.vocab_size)
        test_acc = model.test(args.test_path, save_result="Q1/results/result_bi_gram.json", feature_type="bigram")
        print("Accuracy on test dataset: {:.4f}%".format(100*test_acc))
        model.confusion("Q1/results/result_bi_gram.json", "Q1/plots/confusion_bi_gram.pdf", "Q1/plots/confusion_bi_gram_norm.pdf")
        print(model.compute_f1_class_wise("Q1/results/result_bi_gram.json", save_path="Q1/results/f1_stm_bigram.json"))


        model = NaiveBayesClassifier(alpha=0.1, sw_stm=True)
        model.load_data(args.train_path)
        model.train(freq_thresh_type="hard")
        # model.train(freq_thresh_type="soft")
        print("Training Complete!")
        print("length of vocabulary: ", model.vocab_size)
        test_acc = model.test(args.test_path, save_result="Q1/results/result_freq_thres.json")
        print("Accuracy on test dataset: {:.4f}%".format(100*test_acc))
        model.confusion("Q1/results/result_freq_thres.json", "Q1/plots/confusion_freq_thres.pdf", "Q1/plots/confusion_freq_thres_norm.pdf")

    ### part f
    elif args.part.lower() == "f":
        model = NaiveBayesClassifier(sw_stm=True)
        print(model.compute_f1_class_wise("Q1/results/result_vanilla.json", save_path="Q1/results/f1_vanilla.json"))
        print(model.compute_f1_class_wise("Q1/results/result_majority.json", save_path="Q1/results/f1_majority.json"))
        print(model.compute_f1_class_wise("Q1/results/result_random.json", save_path="Q1/results/f1_random.json"))
        print(model.compute_f1_class_wise("Q1/results/result_stm.json", save_path="Q1/results/f1_stm.json"))
        print(model.compute_f1_class_wise("Q1/results/result_freq_thres.json", save_path="Q1/results/f1_stm_freq_thres.json"))
        print(model.compute_f1_class_wise("Q1/results/result_bi_gram.json", save_path="Q1/results/f1_stm_bigram.json"))

    ### part g
    elif args.part.lower() == "g":
        model = NaiveBayesClassifier(sw_stm=True)
        model.load_data(args.train_path, include_summaries=True)
        model.train()
        print("Training Complete!")
        print("length of vocabulary: ", model.vocab_size)
        test_acc = model.test(args.test_path, save_result="Q1/results/result_summary.json")
        print("Accuracy on test dataset: {:.4f}%".format(100*test_acc))
        model.confusion("Q1/results/result_summary.json", "Q1/plots/confusion_summary.pdf")

    else:
        raise ValueError("The part should be one of a-g, got {}.".format(args.part))


if __name__ == "__main__":
    main()