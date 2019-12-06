from keras.preprocessing.image import ImageDataGenerator

import json
import cv2
import os
import numpy as np

with open('market_attribute.json') as file:
    data = json.load(file)

imdir = 'Market-1501'

attribute_names = [key for key in data['train']][:-1]

train_indexes = data['train']['image_index']
test_indexes = data['test']['image_index']

train_attributes_values = {}
for i, image_index in enumerate(train_indexes):
    train_attributes_values[image_index] = [data['train'][key][i]
                                            for key in attribute_names]

test_attributes_values = {}
for i, image_index in enumerate(test_indexes):
    test_attributes_values[image_index] = [data['test'][key][i]
                                           for key in attribute_names]

images_filenames = os.listdir(imdir)

train_images = []
train_attributes = []
for file in images_filenames:
    image_index = file[0:4]
    if(image_index in train_indexes):
        impath = os.path.join(imdir, file)
        train_images.append(cv2.imread(impath))
        train_attributes.append(train_attributes_values[image_index])
train_images = np.asarray(train_images)
train_attributes = np.asarray(train_attributes)

test_images = []
test_attributes = []
for file in images_filenames:
    image_index = file[0:4]
    if(image_index in test_indexes):
        impath = os.path.join(imdir, file)
        test_images.append(cv2.imread(impath))
        test_attributes.append(test_attributes_values[image_index])
test_images = np.asarray(test_images)
test_attributes = np.asarray(test_attributes)

train_datagen = ImageDataGenerator(rescale=1. / 255.)
train_datagen.fit(train_images)
train_generator = train_datagen.flow(train_images, train_attributes)

test_datagen = ImageDataGenerator(rescale=1. / 255.)
test_datagen.fit(test_images)
test_generator = test_datagen.flow(test_images, test_attributes)
