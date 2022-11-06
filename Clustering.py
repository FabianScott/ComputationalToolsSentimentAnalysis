import os
import time
from itertools import combinations
from Functions import clustering, cluster_closeness_matrix, \
    load_review_vectors, assign_clusters, test_error_clusters


# knn is number of neighbours to use for assigning a label to a new cluster, k is number of clusters
knn, k = 9, 5
n_train, n_test = 2000, 2000
cluster_types = ['K-means', 'Minibatch-Kmeans', 'Agglomerative', 'Birch', 'Spectral']


filename_list = ['test_fasttext', 'test_minhash', 'test_fasttext', 'test_minhash']
path_list = [os.path.join(os.getcwd(), f'{name}_vectors.csv') for name in filename_list]

# Loading Training Data
t = time.time()
ft_train_v, ft_train_r = load_review_vectors(path_list[0], no_reviews=n_train)
mh_train_v, mh_train_r = load_review_vectors(path_list[1], no_reviews=n_train)
print(f'Loading training data took: {time.time() - t}')

# Loading Test data
t = time.time()
ft_test_v, ft_test_r = load_review_vectors(path_list[2], no_reviews=n_test)
mh_test_v, mh_test_r = load_review_vectors(path_list[3], no_reviews=n_test)
print(f'Loading test data took: {time.time() - t}')


proportion_correct, cluster_assignments, cc_mats, models = [], [], [], []
for i, name in enumerate(cluster_types[:1]):
    # Run the clustering
    t = time.time()
    labels_ft, model_ft = clustering(ft_train_v, method=name)
    labels_mh, model_mh = clustering(mh_train_v, method=name)
    print(f'{name} took: {time.time() - t} seconds on both vectorizations')

    # Proportion of each class in the clusters, each row is a cluster column is star rating
    m1 = cluster_closeness_matrix(ft_train_r, labels_ft, decimals=4)
    m2 = cluster_closeness_matrix(mh_train_r, labels_mh, decimals=4)

    # Using the maximum proportions assign each cluster a star rating:
    assigned_ft, assigned_mh = assign_clusters(m1), assign_clusters(m2)

    # Find knn nearest neighbours and vote for cluster to which new point belongs
    correct_proportion_ft = test_error_clusters(ft_test_v, ft_train_v, labels_ft, ft_test_r, knn)
    correct_proportion_mh = test_error_clusters(mh_test_v, mh_train_v, labels_mh, mh_test_r, knn)

    # Append all desired data to corresponding lists
    cluster_assignments.append((assigned_ft, assigned_mh))
    cc_mats.append((m1, m2))
    models.append((model_ft, model_mh))
    proportion_correct.append((correct_proportion_ft, correct_proportion_mh))
print(proportion_correct)
