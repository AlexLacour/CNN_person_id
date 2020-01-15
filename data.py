import json
import cv2
import os
import numpy as np


def get_attributes_and_indexes():
    with open('market_attribute.json') as file:
        data = json.load(file)
    return data


def get_image_indexes(data):
    return data['image_index']


def get_attributes_names():
    data = get_attributes_and_indexes()['train']
    return [key for key in data][:-1]


def get_attributes_values(data):
    indexes = get_image_indexes(data)
    attribute_names = get_attributes_names()

    attributes_values = {}

    for i, image_index in enumerate(indexes):
        attributes_values[image_index] = [data[key][i]
                                          for key in attribute_names]
    return attributes_values


def get_images_attributes(data, imdir='Market-1501'):
    images_filenames = os.listdir(imdir)
    attributes_values = get_attributes_values(data)
    indexes = get_image_indexes(data)

    images = []
    attributes = []

    for file in images_filenames:
        image_index = file[0:4]
        if(image_index in indexes):
            impath = os.path.join(imdir, file)
            images.append(cv2.imread(impath))
            attributes.append(attributes_values[image_index])
    return np.asarray(images), np.asarray(attributes)


def load_data():
    train_data = get_attributes_and_indexes()['train']
    test_data = get_attributes_and_indexes()['test']
    X_train, y_train = get_images_attributes(train_data)
    X_test, y_test = get_images_attributes(test_data)
    return (X_train, y_train), (X_test, y_test)
