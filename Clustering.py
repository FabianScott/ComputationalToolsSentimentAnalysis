import os
import time
import fasttext
import numpy as np
from sklearn.cluster import KMeans
from Functions import k_means, show_clustering, load_review_lists, cluster_closeness_matrix, vectorise_text, minhash_text

# Full path or else error
model_path = 'C:\\Users\\toell\\OneDrive\\Documents\\GitHub\\AZ-sentiment-analysis\\Models\\autotuned_all.bin '

t = time.time()
model = fasttext.load_model(model_path)
data_path = os.path.join(os.getcwd(), 'data')
train_list, test_list = load_review_lists(filepath=data_path)
print(f'Loading data took: {time.time()-t}')
text_vectors = vectorise_text(train_list, model, create_file=True, name='train')
text_minhashed = minhash_text(train_list, create_file=True, name='train')
test_vectors = vectorise_text(test_list, model, create_file=True, name='train')
test_minhashed = minhash_text(test_list, create_file=True, name='train')
print(f'Vectorising data took: {time.time()-t}')

# sklearn clustering
t = time.time()
k = len(np.unique(train_list))
sk_labels_vectors = k_means(text_vectors, homemade=False, show_cluster=True, k=k, title='Sk on Vectors')
sk_labels_minhash = k_means(text_minhashed, homemade=False, show_cluster=True, k=k, title='Sk on Minhash')
print(f'sklearn k_means: {time.time()-t}')

# homemade clustering
t = time.time()
hm_labels_vector = k_means(text_vectors, k=k, homemade=True, show_cluster=True, title='HM on vectors')
hm_labels_minhash = k_means(text_minhashed, k=k, homemade=True, show_cluster=True, title='HM on Minhash')
print(f'own k_means: {time.time()-t}')

# Proportion of each class in the clusters
print(cluster_closeness_matrix(train_list[:, 0], sk_labels_vectors))
print(cluster_closeness_matrix(train_list[:, 0], sk_labels_minhash))
print(cluster_closeness_matrix(train_list[:, 0], hm_labels_vector))
print(cluster_closeness_matrix(train_list[:, 0], hm_labels_minhash))
