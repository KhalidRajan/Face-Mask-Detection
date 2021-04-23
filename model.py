import cv2
import numpy as np
import os
import tensorflow as tf
import keras
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
data_directory=os.listdir("data")
dataset="data"
images=[]
labels=[]

# Resize all images and append images + labels to a list
for item in data_directory:
    folder=os.path.join(dataset, item)
    for image in os.listdir(folder):
        image_path=os.path.join(folder, image)
        image=cv2.imread(image_path)
        resized_image=cv2.resize(image, (150, 150))
        images.append(resized_image)
        labels.append(item)

# Encode the labels

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=np.array(labels)

# Convert the Images list into a NumPy Array
images=np.array(images)/255.0
images=np.reshape(images, (images.shape[0], 150, 150, 3))


# Split the data into train and Test sets
X_train, X_test, y_train, y_test=train_test_split(images, labels, test_size=0.2, random_state=0)

y_train.reshape((-1, 1))
y_test.reshape((-1, 1))

# Building our Model
model=Sequential([
    Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(strides=(2, 2)),

    Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPooling2D(strides=(2,2)),

    Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPooling2D(strides=(2,2)),

    Flatten(),
    Dense(units=128, activation="relu"),
    Dropout(0.3),
    Dense(units=2, activation="softmax")
])

# Implement early stopping
early_stopping=EarlyStopping(monitor="loss", patience=4, restore_best_weights=True, min_delta=0.01)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history=model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

#Save model
model.save("face_detect")