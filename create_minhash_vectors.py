import os
import time
from Functions import load_review_lists, minhash_text

# Full path or else error
t = time.time()
data_path = os.path.join(os.getcwd(), 'data')
train_list, test_list = load_review_lists(filepath=data_path)
print(f'Loading data took: {time.time() - t}')
t = time.time()

minhash_text(test_list, create_file=True, name='testaaaa')
print(f'Vectorising test data took: {time.time() - t}')

minhash_text(train_list, create_file=True, name='trainaaaa')
print(f'Vectorising training data took: {time.time() - t}')
