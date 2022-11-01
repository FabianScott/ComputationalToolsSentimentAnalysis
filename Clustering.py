import os
import time
import fasttext
import numpy as np
from sklearn.cluster import KMeans
from Functions import k_means, show_clustering, load_review_lists, cluster_closeness_matrix, vectorise_text, \
    minhash_text, load_review_vectors

# Full path or else error
model_path = 'C:\\Users\\toell\\OneDrive\\Documents\\GitHub\\AZ-sentiment-analysis\\Models\\autotuned_all.bin '

t = time.time()
model = fasttext.load_model(model_path)
train_minhash_path = os.path.join(os.getcwd(), 'train_minhash_vectors.csv')
train_fasttext_path = os.path.join(os.getcwd(), 'train_fasttext_vectors.csv')
print(f'Loading data took: {time.time() - t}')
text_minhashed, minhash_ratings = load_review_vectors(train_minhash_path)
text_vectors, fasttext_ratings = load_review_vectors(train_fasttext_path)

# sklearn clustering
t = time.time()
k = len(np.unique(minhash_ratings)) + 1
sk_labels_vectors = k_means(text_vectors, homemade=False, show_cluster=True, k=k, title='Sk on Vectors')
sk_labels_minhash = k_means(text_minhashed, homemade=False, show_cluster=True, k=k, title='Sk on Minhash')
print(f'sklearn k_means: {time.time() - t}')

# homemade clustering
t = time.time()
hm_labels_vector = k_means(text_vectors, k=k, homemade=True, show_cluster=True, title='HM on vectors')
hm_labels_minhash = k_means(text_minhashed, k=k, homemade=True, show_cluster=True, title='HM on Minhash')
print(f'own k_means: {time.time() - t}')

# Proportion of each class in the clusters
# print(cluster_closeness_matrix(train_list[:][0], sk_labels_vectors))
# print(cluster_closeness_matrix(train_list[:][0], sk_labels_minhash))
# print(cluster_closeness_matrix(train_list[:][0], hm_labels_vector))
# print(cluster_closeness_matrix(train_list[:][0], hm_labels_minhash))
