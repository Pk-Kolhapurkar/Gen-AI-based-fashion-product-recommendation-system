import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

# Load the precomputed feature vectors and filenames from pickle files
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load ResNet50 model without the top layer for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a Sequential model with ResNet50 and GlobalMaxPooling2D layers
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load and preprocess the query image
img = image.load_img('sample/khade.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Extract features from the query image and normalize
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Initialize NearestNeighbors model and fit with the feature vectors
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

# Find the nearest neighbors (excluding itself)
distances, indices = neighbors.kneighbors([normalized_result])

print(indices)  # Print the indices of nearest neighbors

# Display the nearest neighbor images
for file in indices[0][0:5]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (312, 312)))
    cv2.waitKey(0)
