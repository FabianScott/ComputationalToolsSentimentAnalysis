import os
import pandas as pd
import numpy as np
import random
import time


# Maybe run character grams and word grams
# and use voting system to determine value


def create_text_file(data_in, output_filename='data.txt', split=0.8):
    train_file = open(f'train_{output_filename}', 'a')
    test_file = open(f'test_{output_filename}', 'a')
    length = len(data_in)
    for i, row in enumerate(data_in):
        k = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ! "
        body = ''.lower()
        if i % 10000 == 1:
            print(f'{np.round(i / length * 100, 2)} % done')
        for el in row[1]:
            if el in k:
                body = body + el
        if random.random() < split:
            train_file.write(f'__label__{int(row[0])} {body}')  # required format
            train_file.write('\n')
        else:
            test_file.write(f'__label__{int(row[0])} {body}')  # required format
            test_file.write('\n')
    test_file.close()
    train_file.close()


name_list = [
    'Apparel'
]

t = time.time()
filepath = 'C:\\Users\\toell\\Downloads\\archive'
filename = f'amazon_reviews_us_{name_list[0]}_v1_00.tsv'
full_path = os.path.join(filepath, filename)
data = pd.read_csv(full_path, sep='\t', on_bad_lines='skip')
full_data = data[['star_rating', 'review_body']]
full_data.dropna(inplace=True)
full_data = full_data.values  # now np array
print(f'Time to load: {time.time() - t}')

# full_length = len(full_data)
# class_counts = [np.sum(full_data[0] == i) for i in range(1, 6)]
# print(class_counts)   # min was 368094, we choose 200000
# print(f'Reviews in total: {full_length}')

for i in range(1, 6):
    create_text_file(full_data[full_data[:, 0] == i][:200000], f'class_{i}')
