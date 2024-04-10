import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load pre-trained ResNet50 model

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

# Create a Sequential model with ResNet50 as base and GlobalMaxPooling2D layer

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

# Function to extract features from an image using the model
def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Extract features using the model and normalize the result

    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# List all filenames in the 'images' directory

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

feature_list = []   # List to store extracted features

# Loop through each image file, extract features, and append to the feature_list

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))
# Save the extracted features and filenames to pickle files
pickle.dump(feature_list,open('embeddings.pkl','wb'))   # Save features
pickle.dump(filenames,open('filenames.pkl','wb'))       # save filenames
