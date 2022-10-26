import time
import fasttext
import numpy as np
from sklearn.cluster import KMeans
from Functions import k_means, show_clustering, load_cluster_data, cluster_closeness_matrix

# Full path or else error
model_path = 'C:\\Users\\toell\\OneDrive\\Documents\\GitHub\\Semester_Two_Notes\\ImageAnalysis\\AZ-sentiment-analysis' \
             '\\Models\\autotuned_all.bin '
data_path = 'C:\\Users\\toell\\OneDrive\\Documents\\GitHub\\Semester_Two_Notes\\ImageAnalysis\\AZ-sentiment-analysis' \
            '\\dataset\\test_data1.txt '
t = time.time()
model = fasttext.load_model(model_path)
labels, text_vectors = load_cluster_data(data_path, fasttext_model=model, no_lines=100)
print(f'Loading data: {time.time()-t}')

t = time.time()
k = len(np.unique(labels))
k_means_class = KMeans(k, random_state=0)
k_means_class.fit(text_vectors)
print(f'sklearn k_means: {time.time()-t}')
show_clustering(text_vectors, k_means_class.labels_)
t = time.time()
# hm_clusters = k_means(text_vectors, k=k, show_cluster=True)
print(f'own k_means: {time.time()-t}')

t = time.time()
print(cluster_closeness_matrix(labels, k_means_class.labels_))
print(f'Matrix calculation 1: {time.time() - t}')
t = time.time()
# print(cluster_closeness_matrix(labels, hm_clusters))
# print(f'Matrix calculation 1: {time.time() - t}')
