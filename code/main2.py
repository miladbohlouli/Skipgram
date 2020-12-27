from Doc2vec import *
from mid_interface import *

load = True
saving_path = "dictionary2.pickle"

part1 = doc2vec(name="200-8",
                model_dir="models")

if load:
    saving_dict = dict_load(saving_path)
    train_vecs = saving_dict["train_vecs"]
    test_vecs = saving_dict["test_vecs"]
    train_data = saving_dict["train_data"]
    train_titles = saving_dict["train_titles"]
    test_data = saving_dict["test_data"]
    test_titles = saving_dict["test_titles"]
    final_train_labels = saving_dict["final_train_labels"]
    final_test_labels = saving_dict["final_test_labels"]
    labels_dictionary = saving_dict["labels_dictionary"]

    train_titles_num, labels_dictionary = convert_to_numbers(train_titles)
    test_titles_num,_ = convert_to_numbers(test_titles)

    print("Evaluating the train data")
    doc2vec.evaluation(train_titles_num, final_train_labels)
    print("Evaluating the test data")
    doc2vec.evaluation(test_titles_num, final_test_labels)

else:
    train_data, train_titles, test_data, test_titles = process_data(train_path="corpora/train.txt",
                                                                    test_path="corpora/test.txt")

    part1.train_model(train_path="corpora/processed.txt",
                      num_epochs=200,
                      min_count=10,
                      initial_learning_rate=0.03,
                      window_size=5)

    train_vecs = part1.doc2vec_second_method(train_data)

    # For computational convenience the titles are converted to numbers
    centers, labels = part1.KMeans_clustering(train_vecs)
    train_titles_num, labels_dictionary = convert_to_numbers(train_titles)
    final_train_labels = part1.find_labels_centers(labels, train_titles_num, centers)

    test_vecs = part1.doc2vec_second_method(test_data)
    final_test_labels = part1.find_nearest_center(test_vecs)
    test_titles_num, _ = convert_to_numbers(test_titles)

    print("Evaluating the train data")
    doc2vec.evaluation(train_titles_num, final_train_labels)
    print("Evaluating the test data")
    doc2vec.evaluation(test_titles_num, final_test_labels)

    saving_dict = {"train_vecs": train_vecs,
                   "test_vecs": test_vecs,
                   "train_data": train_data,
                   "train_titles": train_titles,
                   "test_data": test_data,
                   "test_titles": test_titles,
                   "final_train_labels": final_train_labels,
                   "final_test_labels": final_test_labels,
                   "labels_dictionary": labels_dictionary,
                   }

    dict_save(saving_dict, saving_path)
