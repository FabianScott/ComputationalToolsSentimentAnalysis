import time
import fasttext
import numpy as np
from sklearn.cluster import KMeans
from Functions import k_means, show_clustering, load_data, cluster_closeness_matrix, vectorise_text

# Full path or else error
model_path = 'C:\\Users\\toell\\OneDrive\\Documents\\GitHub\\Semester_Two_Notes\\ImageAnalysis\\AZ-sentiment-analysis' \
             '\\Models\\autotuned_all.bin '
data_path = 'C:\\Users\\toell\\OneDrive\\Documents\\GitHub\\Semester_Two_Notes\\ImageAnalysis\\AZ-sentiment-analysis' \
            '\\dataset\\test_data1.txt '
t = time.time()
model = fasttext.load_model(model_path)
train_list, test_list = load_data()
print(f'Loading data: {time.time()-t}')
text_vectors = vectorise_text(train_list, model)

# sklearn clustering
t = time.time()
k = len(np.unique(train_list))
k_means_class = KMeans(k, random_state=0)
k_means_class.fit(text_vectors)
print(f'sklearn k_means: {time.time()-t}')
show_clustering(text_vectors, k_means_class.labels_)

# homemade clustering
t = time.time()
hm_clusters = k_means(text_vectors, k=k, show_cluster=True)
print(f'own k_means: {time.time()-t}')

# Proportion of each class in the clusters
t = time.time()
print(cluster_closeness_matrix(train_list[:, 0], k_means_class.labels_))
print(f'Matrix calculation 1: {time.time() - t}')
t = time.time()
print(cluster_closeness_matrix(train_list[:, 0], hm_clusters))
print(f'Matrix calculation 1: {time.time() - t}')
