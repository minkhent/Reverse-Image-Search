import numpy as np
from numpy.linalg import norm
import pickle
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def load_dataset(dataset_dir):
    image_files = []
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    for root, sub_directories, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                image_files.append(os.path.join(root, filename))
    return image_files

def extract_features(img, model):
    input_shape = (224, 224, 3)
    # load image and convert image to array
    img = image.load_img(img, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    # preprocess image and extract feature using ResNet50 architecture
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    # convert features N-D array to one dimensional array to
    # one-D array and normalized features dividing norm values
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features


# extract image features and save as pickle file
def save_features():
    features_list = []
    for i in range(len(filenames)):
        features_list.append(extract_features(filenames[i], model))
    pickle.dump(features_list, open('data/features-caltech10-resnet.pickle', 'wb'))
    pickle.dump(filenames, open('data/filenames-caltech10.pickle', 'wb'))

# path to the datasets
dataset_dir = 'data/caltech10'
filenames = sorted(load_dataset(dataset_dir))

