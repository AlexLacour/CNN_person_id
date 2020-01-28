import json
import cv2
import os
import numpy as np
from keras.utils import to_categorical


def get_full_data():
    with open('market_attribute.json') as file:
        data = json.load(file)
    return data


def get_images_indexes(data):
    return data['image_index']


def get_attributes_names():
    data = get_full_data()['train']
    return [key for key in data][:-1]


def get_attributes_values(data, preprocess_att=True):
    indexes = get_images_indexes(data)
    attribute_names = get_attributes_names()

    attributes_values = {}

    for i, image_index in enumerate(indexes):
        keys_data = []
        for key in attribute_names:
            key_data = data[key][i]
            if(preprocess_att):
                key_data -= 1
                if(key == 'age'):
                    key_data_age = to_categorical(
                        key_data, num_classes=4, dtype='int')
                    for key_data_age_stat in key_data_age:
                        keys_data.append(key_data_age_stat)
                else:
                    keys_data.append(key_data)
            else:
                keys_data.append(key_data)
        attributes_values[image_index] = keys_data
    return attributes_values


def get_images_attributes(data, imdir='Market-1501', preprocess_att=True):
    images_filenames = os.listdir(imdir)
    attributes_values = get_attributes_values(
        data, preprocess_att=preprocess_att)
    indexes = get_images_indexes(data)

    images = []
    attributes = []

    for file in images_filenames:
        image_index = file[0:4]
        if(image_index in indexes):
            impath = os.path.join(imdir, file)
            images.append(cv2.imread(impath))
            attributes.append(attributes_values[image_index])
    return np.asarray(images), np.asarray(attributes)


def load_att_data(preprocess_att=True):
    train_data = get_full_data()['train']
    test_data = get_full_data()['test']
    X_train, y_train = get_images_attributes(
        train_data, preprocess_att=preprocess_att)
    X_test, y_test = get_images_attributes(
        test_data, preprocess_att=preprocess_att)

    return (X_train, y_train), (X_test, y_test)


def load_ids_data(imdir='Market-1501'):
    train_data = get_full_data()['train']
    test_data = get_full_data()['test']

    train_ids = get_images_indexes(train_data)
    test_ids = get_images_indexes(test_data)

    y_train = []
    y_test = []

    for img_name in os.listdir(imdir):
        image_index = img_name[0:4]
        if(image_index in train_ids):
            img_idx_tmp = int(image_index) - 1
            y_train.append(img_idx_tmp)
        elif(image_index in test_ids):
            img_idx_tmp = int(image_index) - 1
            y_test.append(img_idx_tmp)
    y_train = to_categorical(y_train, num_classes=1501)
    y_test = to_categorical(y_test, num_classes=1501)

    return np.asarray(y_train), np.asarray(y_test)


def data_for_full_model(preprocess_att=False):
    (X_train, y_train), (X_test, y_test) = load_att_data(
        preprocess_att=preprocess_att)
    y_id_train, y_id_test = load_ids_data()

    X_img = []
    y_att = []
    y_id = []

    for img_tr in X_train:
        X_img.append(img_tr)
    for img_te in X_test:
        X_img.append(img_te)

    for att_tr in y_train:
        y_att.append(att_tr)
    for att_te in y_test:
        y_att.append(att_te)

    for id_tr in y_id_train:
        y_id.append(id_tr)
    for id_te in y_id_test:
        y_id.append(id_te)

    X_img = np.asarray(X_img)
    y_att = np.asarray(y_att)
    y_id = np.asarray(y_id)

    return X_img, y_att, y_id


if __name__ == '__main__':
    x, y, z = data_for_full_model(True)
    print(y[0])
