import os
import time
import numpy as np
from itertools import combinations
from Functions import clustering, cluster_closeness_matrix, \
    load_review_vectors, assign_clusters, p_correct_clusters


# knn is number of neighbours to use for assigning a label to a new cluster, k is number of clusters
knn, k = 9, 5
n_train, n_test = 1000, 100
cluster_types = ['K-means', 'Minibatch-Kmeans']     # , 'Gaussian-Mixture' , 'Agglomerative', 'Birch', 'Spectral']


filename_list = ['train_fasttext', 'train_minhash', 'test_fasttext', 'test_minhash']
path_list = [os.path.join(os.getcwd(), f'{name}_vectors.csv') for name in filename_list]

# Loading Training Data
t = time.time()
ft_train_v, ft_train_r = load_review_vectors(path_list[0], no_reviews=n_train)
mh_train_v, mh_train_r = load_review_vectors(path_list[1], no_reviews=n_train)
print(f'Loading training data took: {time.time() - t}')
print(f'Shape of training data:\nft: {ft_train_v.shape}\nmh: {mh_train_v.shape}')
# Loading Test data
t = time.time()
ft_test_v, ft_test_r = load_review_vectors(path_list[2], no_reviews=n_test)
mh_test_v, mh_test_r = load_review_vectors(path_list[3], no_reviews=n_test)
print(f'Loading test data took: {time.time() - t}')
print(f'Shape of test data:\nft: {ft_test_v.shape}\nmh: {mh_test_v.shape}')


proportion_correct, cluster_assignments, cc_mats, models, weights = [], [], [], [], []
for i, name in enumerate(cluster_types):
    # Run the clustering
    t = time.time()
    labels_ft, model_ft = clustering(ft_train_v, method=name)
    print(f'{name} took: {time.time() - t} seconds on fasttext')
    labels_mh, model_mh = clustering(mh_train_v, method=name)
    print(f'{name} took: {time.time() - t} seconds on minhash')

    t = time.time()
    # Proportion of each class in the clusters, each row is a cluster column is star rating
    m1, w1 = cluster_closeness_matrix(ft_train_r, labels_ft, decimals=4)
    m2, w2 = cluster_closeness_matrix(mh_train_r, labels_mh, decimals=4)

    # Using the maximum proportions assign each cluster a star rating, creates a dict:
    label_map_ft, label_map_mh = assign_clusters(m1, w1), assign_clusters(m2, w2)
    # Use predict method and compare to the assigned clusters
    correct_proportion_ft = p_correct_clusters(ft_test_r, ft_test_v, label_map_ft, model=model_ft)
    correct_proportion_mh = p_correct_clusters(mh_test_r, mh_test_v, label_map_mh, model=model_mh)
    print(f'Calculating the correct proportion took {time.time()-t} seconds')

    # Append all desired data to corresponding lists
    cluster_assignments.append((label_map_ft, label_map_mh))
    cc_mats.append((m1, m2))
    models.append((model_ft, model_mh))
    weights.append((w1, w2))
    proportion_correct.append((correct_proportion_ft, correct_proportion_mh))

print(proportion_correct)
print(f'The proportion of each class in each cluster is: {weights}')
