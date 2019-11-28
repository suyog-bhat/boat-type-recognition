
from google.colab import drive
drive.mount('/content/drive')

import tensorflow.keras
from PIL import Image
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
modelggo = tensorflow.keras.models.load_model('./top.h5')

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
DATA_DIR = "." #because already chdir-ed into the  directory 
BATCH_SIZE = 64
img_height, img_width = 224, 224;

Datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split


train_generator = Datagen.flow_from_directory(
    DATA_DIR,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = Datagen.flow_from_directory(
    DATA_DIR,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation') # set as validation data

modelggo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("./top.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto')

import matplotlib.pyplot as plt
classes = list(['cruise ship','buoy', 'ferry boat', 'freight boat', 'gondola', 'inflatable boat', 'kayak', 'paper boat', 'sailboat'])
print(classes)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open('...path/image_name.jpg')
# Make sure to resize all images to 224, 224 otherwise they won't fit in the array
image = image.resize((224, 224))
plt.imshow(image)
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = image_array / 255.0

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = modelggo.predict_classes(data)
print(prediction)
pred=prediction[0]
classes[pred]


