import os
import time
import fasttext
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from Functions import clustering, show_clustering, load_review_lists, cluster_closeness_matrix, vectorise_text, \
    minhash_text, load_review_vectors

# Full path or else error
model_path = 'C:\\Users\\toell\\OneDrive\\Documents\\GitHub\\AZ-sentiment-analysis\\Models\\autotuned_all.bin '

t = time.time()
model = fasttext.load_model(model_path)
train_minhash_path = os.path.join(os.getcwd(), 'test_minhash_vectors.csv')
train_fasttext_path = os.path.join(os.getcwd(), 'test_fasttext_vectors.csv')
print(f'Loading data took: {time.time() - t}')
text_minhashed, minhash_ratings = load_review_vectors(train_minhash_path)
text_vectors, fasttext_ratings = load_review_vectors(train_fasttext_path)

# sklearn clustering
# Implement maxmin initialisation maybe
t = time.time()
k = len(np.unique(minhash_ratings)) + 1
sk_labels_vectors = clustering(text_vectors, homemade=False, show_cluster=True, k=k, title='Sk on Vectors')
sk_labels_minhash = clustering(text_minhashed, homemade=False, show_cluster=True, k=k, title='Sk on Minhash')
print(f'sklearn k_means: {time.time() - t}')

# homemade clustering
t = time.time()
tol = 0.001
# hm_labels_vector = k_means(text_vectors, k=k, homemade=True, show_cluster=True, title='HM on vectors', tol=tol)
# hm_labels_minhash = k_means(text_minhashed, k=k, homemade=True, show_cluster=True, title='HM on Minhash', tol=tol)
print(f'own k_means: {time.time() - t}')

# Proportion of each class in the clusters
print(cluster_closeness_matrix(fasttext_ratings, sk_labels_vectors))
print(cluster_closeness_matrix(minhash_ratings, sk_labels_minhash))
# print(cluster_closeness_matrix(fasttext_ratings, hm_labels_vector))
# print(cluster_closeness_matrix(minhash_ratings, hm_labels_minhash))
