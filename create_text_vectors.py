import os
import time
import fasttext
import numpy as np
from sklearn.cluster import KMeans
from Functions import load_review_lists, vectorise_text, minhash_text

# Full path or else error
t = time.time()
data_path = os.path.join(os.getcwd(), 'data')
train_list, test_list = load_review_lists(filepath=data_path)
model_path = 'C:\\Users\\toell\\OneDrive\\Documents\\GitHub\\AZ-sentiment-analysis\\Models\\autotuned_all.bin '
model = fasttext.load_model(model_path)
print(f'Loading data took: {time.time() - t}')
t = time.time()

vectorise_text(test_list, fasttext_model=model, create_file=True, name='test')
print(f'Vectorising test data took: {time.time() - t}')

vectorise_text(train_list, fasttext_model=model, create_file=True, name='train')
print(f'Vectorising training data took: {time.time() - t}')
