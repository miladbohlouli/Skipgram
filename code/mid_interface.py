import pickle
import numpy as np

#########################################################################
#   A saving tool for a dictionary
#########################################################################
def dict_save(dictionary, path):
    print("Saving the dictionary containing the data")
    file = open(path, "wb")
    pickle.dump(dictionary, file)
    file.close()


#########################################################################
#   A loading tool for a dictionary
#########################################################################
def dict_load(path):
    print("Loading the dictionary containing the data")
    file = open(path, "rb")
    obj = pickle.load(file)
    file.close()
    return obj


def process_data(train_path, test_path):
    print("processing the train and the test data")
    train_file = open(train_path, "r", encoding="utf-8")
    test_file = open(test_path, "r", encoding="utf-8")
    temp_file = open("corpora/processed.txt", "a", encoding="utf-8")

    line = train_file.readline()
    train_documents = []
    test_documents = []
    train_titles = []
    test_titles = []

    print("Importing the train data")
    while line:
        title, _, text = line.partition("@@@@@@@@@@")
        train_titles.append(title)
        train_documents.append(text)
        temp_file.writelines(text)
        line = train_file.readline()

    line = test_file.readline()
    print("Importing the test data")
    while line:
        title, _, text = line.partition("@@@@@@@@@@")
        test_titles.append(title)
        test_documents.append(text)
        temp_file.writelines(text)
        line = test_file.readline()

    return train_documents, train_titles, test_documents, test_titles


def convert_to_numbers(text_titles):
    titles = np.zeros_like(text_titles, dtype=np.int) - 1
    text_titles = np.asarray(text_titles)
    label_dictionary = dict()
    for i, label in enumerate(np.unique(text_titles)):
        titles[text_titles == label] = i
        label_dictionary[i] = label
    return titles, label_dictionary
