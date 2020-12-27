import nltk
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import time
import pickle
import multiprocessing
import logging
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, f1_score, v_measure_score
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
cores = multiprocessing.cpu_count()


#################################################################################
#   In this class train_path stands for the location of the train data
#       and model_dir stands for the place that the model have been of should be
#       saved.
#################################################################################
class doc2vec:
    ###########################################################################################
    #   In this method the model is initialized but at first the availability of
    #       a previous model is checked, if available it will be loaded,
    #       otherwise it will be trained with the specified training data and
    #       will be saved in the model_dir directory.
    #       parameters:
    #           model_dir: path that the model(The whole object will be saved or restored from)
    #           name: The name that the model will be specified with and also saved with
    ##############################################################################################
    def __init__(self, name, model_dir):
        self.name = name
        self.path = model_dir + "/word2vec(%s).pickle" % self.name
        self.model = None
        self.vocabulary = []
        self.centers = dict()

        temp = doc2vec.load_obj(self.path)
        if not temp:
            print("No saved model")

        else:
            self.model = temp.model
            self.vocabulary = temp.vocabulary
            self.centers = temp.centers

        if self.model is not None:
            sample_word = self.vocabulary[2]
            print(self.model.wv.most_similar(positive=sample_word))
            print(sample_word)

    def __del__(self):
        try:
            os.remove("corpora/processed.txt")
        except:
            pass

    ##########################################################################
    #   A method that loads a saved model from the specified model_dir
    ##########################################################################
    @classmethod
    def load_obj(cls, path):
        try:
            file = open(path, "rb")
            print("Attempting to load the model if available from path:\n%s" % path)
            loaded_object = pickle.load(file)
            print("Last checkpoint loaded")
            file.close()
            return loaded_object

        except:
            return False

    @classmethod
    def save_obj(cls, obj, path):
        print("Saving the model in:\n%s" % path)
        file = open(path, "wb")
        pickle.dump(obj, file)
        file.close()

    ##########################################################################
    #   Creates a skipgram model and also creates a vocabulary of the
    #       documents.
    #       train_path: path of the training data
    #       num_epochs: number of the training epochs
    #       min_count: minimum number of occurrences that the words will enter the vocabulary
    #       initial_learning_rate: the starting leaning rate
    #       window_size: The size of the moving window in skipgram model
    ##########################################################################
    def train_model(self, train_path, num_epochs, min_count, initial_learning_rate, window_size):
        if self.model is None:
            print("Training a new model")
            start = time.time()
            Sentence = LineSentence(train_path)
            self.model = Word2Vec(min_count=min_count,
                                  alpha=initial_learning_rate,
                                  min_alpha=0.0007,
                                  negative=20,
                                  size=300,
                                  window=window_size,
                                  workers=cores)

            self.model.build_vocab(corpus_file=Sentence)
            self.model.train(corpus_file=Sentence, total_examples=1, epochs=num_epochs)
            print("The training process took %.2f minutes" % ((time.time() - start) / 60))
            self.vocabulary = [k for k in self.model.wv.vocab]
            doc2vec.save_obj(self, self.path)

    ##########################################################################
    #   In each of the following methods, one method is implemented for
    #       document representation using vector representation of the words
    ##########################################################################
    #   This method simply takes an average over all the words of the
    #       document that are present in our vocabulary
    ##########################################################################
    def doc2vec_first_method(self, data):
        print("calculating the vector representations of the documents")
        start = time.time()
        docs_vector = []
        for i, doc in enumerate(data):
            if i % 100 == 0:
                print("Converting the documents to vectors(%d%%)" % (i / len(data) * 100))
            temp_vector = 0
            word_count = 0
            tokens = nltk.word_tokenize(doc)
            for token in tokens:
                if token in self.vocabulary:
                    temp_vector += self.model.wv[token]
                    word_count += 1
            temp_vector /= word_count
            docs_vector.append(temp_vector)

        print("Done calculation in %.2f minutes" % ((time.time() - start) / 60))
        return docs_vector

    ##########################################################################
    #   This method takes a weighted average over all of the words of the
    #       document that are present in the vocabulary. The weights are
    #       the tf-idf avlues of each word according to the training corpus
    ##########################################################################
    def doc2vec_second_method(self, data):
        print("calculating the vector representations of the documents")
        start = time.time()
        vectorizer = TfidfVectorizer(encoding="utf8")
        doc_term = vectorizer.fit_transform(data)
        docs_vector = []
        for i, doc in enumerate(data):
            if i % 100 == 0:
                print("Converting the documents to vectors(%d%%)" % (i / len(data) * 100))
            temp_vector = 0
            word_count = 0
            tf_idf_sum = 0
            tokens = nltk.word_tokenize(doc)
            for token in tokens:
                if token in self.vocabulary and token in vectorizer.vocabulary_:
                    tf_idf_w = doc_term[i, vectorizer.vocabulary_[token]]
                    temp_vector += self.model.wv[token] * tf_idf_w
                    tf_idf_sum += tf_idf_w
                    word_count += 1
            temp_vector /= word_count
            temp_vector /= tf_idf_sum
            docs_vector.append(temp_vector)
        print("Done calculation in %.2f minutes" % ((time.time() - start) / 60))
        return docs_vector

    ##########################################################################
    #   In this method the vector of each document is calculated using the
    #       doc2vec method of the gensim library
    #       train_path: path of the training data
    #       num_epochs: number of the training epochs
    #       min_count: minimum number of occurrences that the words will enter the vocabulary
    #       initial_learning_rate: the starting leaning rate
    #       window_size: The size of the moving window in skipgram model
    ##########################################################################
    def doc2vec_third_method(self, train_path, initial_learning_rate, window_size, num_epochs, data=None):
        if self.model is None:
            print("Training the doc2vec representations using gensim models\n "
                  "No need to call the train_model function individually")
            start = time.time()
            self.model = Doc2Vec(alpha=initial_learning_rate,
                                 corpus_file=train_path,
                                 min_alpha=0.007,
                                 negative=20,
                                 vector_size=300,
                                 window=window_size,
                                 workers=cores,
                                 epochs=num_epochs)

            print("The training process took %.2f minutes" % ((time.time() - start) / 60))
            self.vocabulary = [k for k in self.model.wv.vocab]
            doc2vec.save_obj(self, self.path)

        print("calculating the vector representations of the documents")
        start = time.time()
        docs_vector = []
        for i, doc in enumerate(data):
            if i % 100 == 0:
                print("Converting the documents to vectors(%d%%)" % (i / len(data) * 100))
            tokens = nltk.word_tokenize(doc)
            true_tokens = []
            for token in tokens:
                if token in self.vocabulary:
                    true_tokens.append(token)
            vector = self.model.infer_vector(true_tokens)
            docs_vector.append(vector)

        print("Done calculation in %.2f minutes" % ((time.time() - start) / 60))
        return docs_vector

    ##########################################################################
    #   This function counts the number of the words on each document type
    #       and calculates the term-document matrix of the corpus, then the
    #       tf-idf values of the words according to the documents are calculated.
    #       Finally the reduced vector representation of the documents are
    #       calculated using SVD(Singular Value Decomposition)
    ##########################################################################
    def doc2vec_forth_method(self, data):
        print("calculating the vector representations of the documents")
        start = time.time()
        vectorizer = TfidfVectorizer(encoding="utf8")
        doc_term = vectorizer.fit_transform(data)
        doc_term = doc_term.A
        svd = TruncatedSVD(n_components=300)
        svd.fit(doc_term)
        u = svd.transform(doc_term)


        # u, sigma, VT = randomized_svd(doc_term,
        #                               n_components=300,
        #                               n_iter=100,
        #                               random_state=20)

        # u, s, vh = np.linalg.svd(doc_term, full_matrices=False)
        print(u.shape)
        summarized_doc_term = u[:, 0:300]
        print("Done calculation in %.2f in minutes" % ((time.time() - start) / 60))
        return summarized_doc_term

    ##########################################################################
    #   In the previous methods we represented each document with a vector
    #       but in this part our goal is clustering the prior vectors
    #       using Kmeans.
    ##########################################################################
    def KMeans_clustering(self, vectors):
        print("Applying the Kmeans clustering")
        vectors = np.array(vectors)
        km = KMeans(n_clusters=5).fit(vectors)
        return km.cluster_centers_, km.labels_


    ##########################################################################
    #   This function simply takes the output of the KMeans algorithm and
    #       assigns the label of each cluster to the most frequent title
    #       among the documents of that cluster.
    ##########################################################################
    def find_labels_centers(self, kmeans_labels, true_labels, centers):
        final_labels = np.zeros_like(kmeans_labels) - 1
        for pred_label in np.unique(kmeans_labels):
            max_count = 0
            label1 = None
            for true_label in np.unique(true_labels):
                count = np.sum(true_labels[kmeans_labels == pred_label] == true_label)
                if count > max_count:
                    label1 = true_label
                    max_count = count
            self.centers[label1] = centers[pred_label]
            final_labels[kmeans_labels == pred_label] = label1
        doc2vec.save_obj(self, self.path)
        return final_labels

    ##########################################################################
    #   This is simply for predicting the nearest center to the vectors
    ##########################################################################
    def find_nearest_center(self, doc_vecs):
        doc_vecs = np.asarray(doc_vecs)
        centers = np.asarray([v for v in self.centers.values()])
        labels = np.asarray([k for k in self.centers.keys()])

        cosine_distances = np.matmul(doc_vecs, np.transpose(centers))
        pred_labels = np.argmax(cosine_distances, axis=1)
        for i in range(len(pred_labels)):
            pred_labels[i] = labels[pred_labels[i]]

        return pred_labels


    @staticmethod
    def evaluation(ground_truth, pred_labels):
        assert len(ground_truth) == len(pred_labels)
        NMI = normalized_mutual_info_score(ground_truth, pred_labels)
        Accuracy = accuracy_score(ground_truth, pred_labels)
        f1 = f1_score(ground_truth, pred_labels, average="macro")
        v = v_measure_score(ground_truth, pred_labels)
        print("The evaluation metrices:\nNMI:%.2f\tAccuracy:%.2f\nf1_score:%.2f\tV_measure:%.2f" % (NMI, Accuracy, f1, v))
        return NMI, Accuracy, f1, v

