import json
import cv2
import os
import numpy as np


def get_full_data():
    with open('market_attribute.json') as file:
        data = json.load(file)
    return data


def get_images_indexes(data):
    return data['image_index']


def get_attributes_names():
    data = get_full_data()['train']
    return [key for key in data][:-1]


def get_attributes_values(data):
    indexes = get_images_indexes(data)
    attribute_names = get_attributes_names()

    attributes_values = {}

    for i, image_index in enumerate(indexes):
        attributes_values[image_index] = [data[key][i]
                                          for key in attribute_names]
    return attributes_values


def get_images_attributes(data, imdir='Market-1501'):
    images_filenames = os.listdir(imdir)
    attributes_values = get_attributes_values(data)
    indexes = get_images_indexes(data)

    images = []
    attributes = []

    for file in images_filenames:
        image_index = file[0:4]
        if(image_index in indexes):
            impath = os.path.join(imdir, file)
            images.append(cv2.imread(impath))
            attributes.append(attributes_values[image_index])
    return np.asarray(images, dtype='float32'), np.asarray(attributes, dtype='float32')


def load_att_data():
    train_data = get_full_data()['train']
    test_data = get_full_data()['test']
    X_train, y_train = get_images_attributes(train_data)
    X_test, y_test = get_images_attributes(test_data)

    y_train[:, :] -= 1
    y_test[:, :] -= 1
    y_train[:, 0] *= (1. / 3.)
    y_test[:, 0] *= (1. / 3.)

    return (X_train, y_train), (X_test, y_test)


def load_ids_data():
    train_data = get_full_data()['train']
    test_data = get_full_data()['test']

    ids_train = get_images_indexes(train_data)
    ids_test = get_images_indexes(test_data)

    return (ids_train, ids_test)


if __name__ == '__main__':
    print(get_attributes_names())
    (_, y), (_, _) = load_att_data()
    for el in y[0:5]:
        print(el)
