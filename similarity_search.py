from sklearn.neighbors import NearestNeighbors
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import time
from feature_extractor import extract_features, model

start_time = time.time()

def k_nearest_neighborhood(features):
    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine').fit(feature_list)
    cosine_distance, indices = neighbors.kneighbors([features])
    return cosine_distance, indices


def get_similar_image(indices, img_path):
    # convert np array to integer scalar array
    indices_arr = numpy.array(indices[0]).astype(int)
    plt.imshow(mpimg.imread(img_path))
    plt.show()
    fig = plt.figure(figsize=(10,10))
    for i in range(1,10):
        img = mpimg.imread(filenames[indices_arr[i-1]])
        fig.add_subplot(5,3,i)
        plt.imshow(img)
    plt.show()


img_path = 'image/18.jpg'
extracted_features = extract_features(img_path, model)

filenames = pickle.load(open('data/filenames-caltech10.pickle', 'rb'))
feature_list = pickle.load(open('data/features-caltech10-resnet.pickle', 'rb'))

cosine_distance, indices = k_nearest_neighborhood(extracted_features)
similar_image = get_similar_image(indices, img_path)

print('-----%s seconds-----'%(time.time() - start_time))

