import os
import numpy as np
import pandas as pd
from Functions import load_review_vectors

filepath = os.path.join(os.getcwd(), 'test_fasttext_vectors.csv')
# minhash_vectors, ratings = load_review_vectors(filepath)

df = pd.read_csv(filepath)
output_vectors = df.values[:, 1:]
if type(output_vectors[0][0]) == str:
    temp = []
    for vector in output_vectors:
        temp.append(np.array(vector[0].strip('][').split(), dtype=float))
    output_vectors = np.array(temp)
ratings = df.values[:, 0]
