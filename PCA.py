import pickle
from sklearn.decomposition import PCA

num_feature_dimension = 100
filenames = pickle.load(open('data/filenames-caltech10.pickle', 'rb'))
feature_list = pickle.load(open('data/features-caltech10-resnet.pickle', 'rb'))

pca = PCA(n_components = num_feature_dimension)
pca.fit(feature_list)

compressed_feature_list = pca.transform(feature_list)
print(len(compressed_feature_list))
print(compressed_feature_list.shape)




