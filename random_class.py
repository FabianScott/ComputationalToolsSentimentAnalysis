import os
import time
import numpy as np
from Functions import load_review_vectors
filename_list = ['train_fasttext', 'train_minhash', 'test_fasttext', 'test_minhash']
path_list = [os.path.join(os.getcwd(), f'{name}_vectors.csv') for name in filename_list]

n_test = None

# Loading Test data
t = time.time()
ft_test_v, ft_test_r = load_review_vectors(path_list[2], no_reviews=n_test)
mh_test_v, mh_test_r = load_review_vectors(path_list[3], no_reviews=n_test)
print(f'Loading test data took: {time.time() - t}')
print(f'Shape of test data:\nft: {ft_test_v.shape}\nmh: {mh_test_v.shape}')

np.random.seed(42)
ft_true = sum((ft_test_r - np.random.randint(1, 6, size=ft_test_r.shape)) == 0)/len(ft_test_r)
mh_true = sum((mh_test_r - np.random.randint(1, 6, size=mh_test_r.shape)) == 0)/len(mh_test_r)
print(f'Proportion correct on Fasttext: {ft_true}')
print(f'Proportion correct on Minhash:  {mh_true}')
