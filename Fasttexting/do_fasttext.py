import fasttext
import numpy as np
import os
import pandas as pd

model = fasttext.train_supervised('train_fasttext.txt', autotuneValidationFile='test_fasttext.txt', autotuneDuration=600)
model.save_model('autotuned_apparel.bin')

create_concat_files = False
if create_concat_files:
    filenames = [os.path.join(os.getcwd(), f'data/train_class_{i}') for i in range(1, 6)]

    with open("train_fasttext_vectors.txt", "w") as outfile:
        for filename in filenames:
            with open(filename) as infile:
                contents = infile.read()
                outfile.write(contents)

    filenames1 = [os.path.join(os.getcwd(), f'data/test_class_{i}') for i in range(1, 6)]
    with open("test_fasttext_vectors.txt", "w") as outfile:
        for filename in filenames1:
            with open(filename) as infile:
                contents = infile.read()
                outfile.write(contents)
