import os
import time
import fasttext
import numpy as np
from sklearn.cluster import KMeans
from Functions import k_means, show_clustering, load_review_lists, cluster_closeness_matrix, vectorise_text, \
    minhash_text

# Full path or else error
model_path = 'C:\\Users\\toell\\OneDrive\\Documents\\GitHub\\AZ-sentiment-analysis\\Models\\autotuned_all.bin '

t = time.time()
model = fasttext.load_model(model_path)
data_path = os.path.join(os.getcwd(), 'data')
train_list, test_list = [
                            ["__label__2", "i ordered the same size as i ordered last time and these shirts were much "
                                           "larger than the previous order they were also about 6 inches longer it "
                                           "was like they "
                                           "sent mens shirts instead of boys shirts ill be returning these"],
                            ["__label__2", "cute dress  please be advised that this cute dress runs very very small in "
                                           "size you would definitely need to order a couple sizes or even three "
                                           "sizes larger than "
                                           "normal  im 5 34334 it is much shorter than i expected also very sheer you "
                                           "would "
                                           "definitely  have to wear a slip with it  this would definitely be a cute "
                                           "dress to wear "
                                           "if i could have fit it"],
                            ["__label__2",
                             "the pants were great! just one thingit wasnt the right color apparently grey "
                             "means khaki but thats none of my businessbr in all seriousness the fit was great but i "
                             "would have loved to have gotten the right color"],
                            ["__label__2",
                             "the dress is cute but the top runs small and the bottom is very longim going "
                             "to cut"
                             "                         off the top and make it into a skirt"],
                            ["__label__2", "thats not the right house that picture is the kraken of house greyjoy"],
                            ["__label__1", "no like picture  looks cheap"]
                        ], []  # load_review_lists(filepath=data_path)

print(f'Loading data took: {time.time() - t}')
text_vectors = vectorise_text(train_list, model, create_file=True, name='train')
text_minhashed = minhash_text(train_list, create_file=True, name='train')
test_vectors = vectorise_text(test_list, model, create_file=True, name='test')
test_minhashed = minhash_text(test_list, create_file=True, name='test')
print(f'Vectorising data took: {time.time() - t}')

# sklearn clustering
t = time.time()
k = len(np.unique([el[0] for el in train_list]))
sk_labels_vectors = k_means(text_vectors, homemade=False, show_cluster=True, k=k, title='Sk on Vectors')
sk_labels_minhash = k_means(text_minhashed, homemade=False, show_cluster=True, k=k, title='Sk on Minhash')
print(f'sklearn k_means: {time.time() - t}')

# homemade clustering
t = time.time()
hm_labels_vector = k_means(text_vectors, k=k, homemade=True, show_cluster=True, title='HM on vectors')
hm_labels_minhash = k_means(text_minhashed, k=k, homemade=True, show_cluster=True, title='HM on Minhash')
print(f'own k_means: {time.time() - t}')

# Proportion of each class in the clusters
print(cluster_closeness_matrix(train_list[:][0], sk_labels_vectors))
print(cluster_closeness_matrix(train_list[:][0], sk_labels_minhash))
print(cluster_closeness_matrix(train_list[:][0], hm_labels_vector))
print(cluster_closeness_matrix(train_list[:][0], hm_labels_minhash))
