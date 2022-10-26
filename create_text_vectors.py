import os
import time
import fasttext
import numpy as np
from sklearn.cluster import KMeans
from Functions import k_means, show_clustering, load_review_lists, cluster_closeness_matrix, vectorise_text, \
    minhash_text

# Full path or else error
t = time.time()
data_path = os.path.join(os.getcwd(), 'data')
train_list, test_list = load_review_lists(filepath=data_path)

print(f'Loading data took: {time.time() - t}')
text_minhashed = minhash_text(train_list, create_file=True, name='train')
test_minhashed = minhash_text(test_list, create_file=True, name='test')
print(f'Vectorising data took: {time.time() - t}')
