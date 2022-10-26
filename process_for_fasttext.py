import os
import pandas as pd
import numpy as np
import random


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
            print(f'{i / length * 100} % done')
        for el in row[1]:
            if el in k:
                body = body + el
        if random.random() < split:
            train_file.write(f'__label__{row[0]} {body}')  # required format
            train_file.write('\n')
        else:
            test_file.write(f'__label__{row[0]} {body}')  # required format
            test_file.write('\n')


name_list = [
    'Apparel'
]

filepath = 'C:\\Users\\toell\\Downloads\\archive'

filename = f'amazon_reviews_us_{name_list[0]}_v1_00.tsv'
full_path = os.path.join(filepath, filename)
data = pd.read_csv(full_path, sep='\t', on_bad_lines='skip')
full_data = data[['star_rating', 'review_body']]
full_data.dropna(inplace=True)
full_data = full_data.values  # now np array
# create_text_file(full_data[:100], output_filename='datasample.txt')


for name in name_list[1:]:
    filename = f'amazon_reviews_us_{name}_v1_00.tsv'
    full_path = os.path.join(filepath, filename)
    data = pd.read_csv(full_path, sep='\t', on_bad_lines='skip')
    relevant_columns = data[['star_rating', 'review_body']]
    relevant_columns.dropna(inplace=True)
    full_data = np.vstack((full_data, relevant_columns.values))  # np array

full_length = len(full_data)
print(f'Reviews in total: {full_length}')
create_text_file(full_data, output_filename='data1.txt')
print(f'Reviews in total: {full_length}')
