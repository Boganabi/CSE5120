# Convolutional Neural Network for Binary cat/dog classification

# Import required libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
#%%
# Training data preprocessing - Keras preprocessing library
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range  = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True)
training_set = training_datagen.flow_from_directory('dataset/training_set',
                                                    target_size = (64,64),
                                                    batch_size  = 32,
                                                    class_mode='binary')

# Test data preprocessing - Keras preprocessing library
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (64,64),
                                                    batch_size  = 32,
                                                    class_mode='binary')
#%%
# Develop CNN model

# Initialization 
cnn = tf.keras.models.Sequential()

# Step 2 Convolution

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))

# Step 3 Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 2 Convolution

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# Step 3 Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 4 Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 5 Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 6 Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile our CNN
cnn.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Train our model
cnn.fit(x = training_set, validation_data=test_set, epochs = 25)

# Run the model on test image

#%% 
import tensorflow as tf
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded model from disk")


#%%
# serialize model to JSON
model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn.save_weights("model.h5")
print("Saved model to disk")
 
# later...
#%% 
import tensorflow as tf
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded model from disk")


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)