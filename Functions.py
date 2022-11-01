import os
import random

import mmh3
import numpy as np
from copy import copy
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN


def read_fastext_files(filepath):
    output = []
    with open(filepath) as file:
        output.append(file.readline())
    return output


def load_review_lists(filepath, no_reviews=np.inf):
    """
    Given a filepath to the folder storing all the reviews, this returns
    a np array of the training data, with stars as the first column, reviews
    as second and the same for the test data. Returns the arrays shuffled.
    """
    train_list, test_list = [], []
    counter = 0
    for i in range(1, 6):
        filepath1 = os.path.join(filepath, f'train_class_{i}')
        with open(filepath1) as file:
            for line in file.readlines():
                if counter >= no_reviews:
                    break
                if len(line.split()) == 1:  # Quick fix for no review
                    line = [line[9], ' ']
                    train_list.append(line)
                else:
                    train_list.append([line[9], line[11:-1]])
                counter += 1
            if counter > no_reviews:
                break
        filepath2 = os.path.join(filepath, f'test_class_{i}')
        with open(filepath2) as file:
            for line in file.readlines():
                if counter > no_reviews:
                    break
                if len(line.split()) == 1:  # Quick fix for no review
                    line = [line[9], ' ']
                    train_list.append(line)
                else:
                    test_list.append([line[9], line[11:-1]])
                counter += 1
            if counter > no_reviews:
                break
    np.seed(2)
    np.random.shuffle(train_list)
    np.random.shuffle(test_list)
    return train_list, test_list


def load_review_vectors(filepath):
    """
    Given the filepath to a csv file containing the vector representation
    of texts return these as np arrays, including the ratings
    :param filepath:
    :return:
    """
    df = pd.read_csv(filepath)
    output_vectors = df.values[:, 1:]
    if type(output_vectors[0][0]) == str:
        temp = []
        for vector in output_vectors:
            temp.append(np.array(vector[0].strip('][').split(), dtype=float))
        output_vectors = np.array(temp)
    ratings = df.values[:, 0]

    return output_vectors, ratings


def vectorise_text(review_list, fasttext_model, create_file=False, name='Train'):
    """
    Given a list of ratings and reviews and a fasttext model, this function
    returns the average of the word vectors as represented by the model.
    """
    output = []
    counter = 0
    for review in review_list:
        if counter % 1000 == 1:
            print(f'{np.round(100*counter/len(review_list), 6)} % done')
        text_list = []
        for word in review[1]:
            temp_vector = fasttext_model.get_word_vector(word)
            text_list.append(temp_vector)
        output.append(np.sum(np.array(text_list), axis=0) / len(text_list))
        counter += 1
    output = [[review_list[i][0], el] for i, el in enumerate(output)]
    if create_file:
        df = pd.DataFrame(output)
        df.to_csv(f'{name}_fasttext_vectors.csv', header=True, index=False)
    return output


def minhash_text(review_list, q=9, seed=0, minhash_length=100, create_file=False, name=''):
    """
    Given the review list including the star ratings, return a list
    of each review minhashed. The seed and length of minhash vector
    can be specified.
    
    :param name:
    :param create_file:
    :param review_list:         contains [stars, review_text]
    :param q: 
    :param minhash_length: 
    :param seed:     
    :return: 
    """
    output = []
    counter = 0
    for review in review_list:
        if counter % 1000 == 1:
            print(f'{np.round(100*counter/len(review_list), 6)} % done')
        shingles_list = q_shingles(review[1], q)
        output.append(minhash(shingles_list, k=minhash_length, stars=int(review[0])))
        counter += 1
    if create_file:
        df = pd.DataFrame(output, dtype=int)
        df.to_csv(f'{name}_minhash_vectors.csv', header=True, index=False)
    return output


def minhash(shingles_list, k=1, stars=None):
    """
    Given list of shingles of words representing a text, compute the
    list of min-hashes of length k for that text, can add star rating
    as the first column if specified
    :param stars:
    :param shingles_list:
    :param k:
    :return minh:
    """
    minh = []
    if stars is not None:
        minh.append(stars)

    for i in range(k):  # define the seeds for the hash function
        temp = np.inf
        if shingles_list:
            for shingle in shingles_list:
                temp = min(temp, mmh3.hash(shingle, seed=i, signed=False))
            minh.append(int(temp))
        else:
            minh.append(0)
    return minh


def q_shingles(string, q, characters=True):
    """
    Given string and length of shingles, returns a set of
    all shingles of characters/words in the string.
    :param characters:
    :param string:
    :param q:
    :return output:
    """
    output = set()
    if characters:
        length = len(string)
        for i in range(length-q):
            output.add(string[i:i+q])
    else:
        string_list = string.split()
        length = len(string_list)
        for i in range(length - q):
            output.add(tuple(string[i:i + q]))
    return output


def set_similarity(s1, s2, metric='jac'):
    """
    Given the name of the similarity measure, compute it for
    the two given list/sets s1 and s2. The options are:
    Jaccard
    Sørensen
    Overlap
    This function only needs the first letter in the name.
    :param s1:
    :param s2:
    :param metric:
    :return: similarity measure
    """

    metric = metric[:1].lower()
    output = 0
    s1, s2 = set(s1), set(s2)
    if metric == 'j':
        output = len(s1.intersection(s2))/len(s1.union(s2))
    elif metric == 's':
        output = 2*len(s1.intersection(s2))/(len(s1) + len(s2))
    elif metric == 'o':
        output = len(s1.intersection(s2))/min(len(s1), len(s2))

    return output


def k_means(points, homemade=False, k=10, centroids=None, tol=1e-5, show_cluster=False, title='Clusters Shown'):
    """
    K-means clustering using either the homemade implementation or sk-learn's.
    It can show a 2D plot of the first two dimensions of the clusters.
    In the homemade version the centroids can be specified.
    :param points:
    :param homemade:
    :param k:
    :param centroids:
    :param tol:
    :param show_cluster:
    :param title:
    :return:
    """
    points = np.array(points)
    if homemade:
        if centroids is None:
            random_indices = np.random.randint(0, len(points), k)
            centroids = np.array([points[i] for i in random_indices])

        cluster_assignments = np.zeros(len(points), dtype=int)
        temp = np.ones(centroids.shape) - centroids
        while np.array([el > tol for el in np.abs(centroids - temp)]).any():
            temp = copy(centroids)
            # cluster_assignments = [np.argmin(np.sum(np.abs(centroids - point), axis=1)) for i, point in enumerate(points)]
            for i, point in enumerate(points):
                distances = np.sum(np.abs(centroids - point), axis=1)
                cluster_assignments[i] = int(np.argmin(distances))
            for i in range(k):
                ci_points = points[cluster_assignments == i]
                centroids[i] = np.sum(ci_points, axis=0) / len(ci_points)
    else:
        k_means_class = KMeans(k, random_state=0)
        k_means_class.fit(points)
        cluster_assignments = k_means_class.labels_

    if show_cluster:
        show_clustering(points, cluster_assignments, title=title)

    return cluster_assignments


def show_clustering(points, assignments, title='Clusters shown'):
    points = np.array(points)
    if points.shape[1] > 2:
        print('Show_cluster will only show the first 2 dimensions of points!!')
    cmap = {0: 'b', 1: 'y', 2: 'g', 3: 'r', 4: 'm', 5: 'c', 6: 'k'}
    try:
        color_list = [cmap[el] for el in assignments]
    except ValueError:
        print('Only 7 colours are available, plot failed')
        raise ValueError
    plt.scatter(points[:, 0], points[:, 1], c=color_list)
    plt.title(title)
    plt.show()


def cluster_closeness_matrix(true_labels, clusters, decimals=3):
    """
    Calculate the percentage of labels corresponding to each true label
    in each cluster.
    """
    k = len(np.unique(true_labels))
    percent_chance = []
    for i in range(k):
        counts = [np.sum(true_labels[clusters == i] == j) for j in range(1, k + 1)]
        if any(counts):
            percent_chance.append(counts / sum(counts))
    percent_chance = np.round(np.array(percent_chance), decimals)
    return percent_chance


def jaccard_estimate(doc1, doc2, q=9, k=100):
    """
    Given filepaths to 2 documents, compute the estimated Jaccard similarity

    :param doc1:
    :param doc2:
    :param q:
    :param k:
    :return: jac
    """
    with open(doc1) as f:
        d1 = f.read()
    with open(doc2) as f:
        d2 = f.read()

    s1 = q_shingles(d1, q)
    s2 = q_shingles(d2, q)
    print(len(s1.intersection(s2))/len(s1.union(s2)))
    s1 = set(s1)
    s2 = set(s2)
    m1 = set(minhash(s1, k=k))
    m2 = set(minhash(s2, k=k))

    jac = len(m1.intersection(m2))/len(m1.union(m2))

    return jac


def similar(doc_name_list, q=9, k=100):
    doc_pairs = set()
    output = []
    for doc in doc_name_list:
        for other_doc in doc_name_list:
            doc_pairs.add((doc, other_doc))
    for pair in doc_pairs:
        if jaccard_estimate(pair[0], pair[1], q=q, k=k) >= 0.6:
            output.append(pair)
    return output


def other_jac(doc_names, q=9, k=100):
    Sig = [[np.inf for j in range(k)] for i in range(len(doc_names))]
    U = set()
    for doc in doc_names:
        with open(doc) as f:
            text = f.read()
            U.add(q_shingles(text, q=q))

    for string in U:
        string_hash = minhash([string], k=k)    # Returns a list
        for i, row in enumerate(Sig):           # Iterates through documents
            for j in range(k):                  # Iterates through hashes
                Sig[i][j] = min(Sig[i][j], string_hash[0])
    return Sig
