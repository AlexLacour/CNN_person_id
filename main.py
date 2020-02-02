from keras.models import load_model, Model
import numpy as np
import train
import os
import cv2
import data


def euclidian_distance(y1, y2):
    return np.sqrt(np.sum(np.square([y1_el - y2_el for y1_el, y2_el in zip(y1, y2)]), axis=-1))


def img_features_comparison(img1, img2, model):
    img_features_layer = model.get_layer('img_features')
    features_getter = Model(
        inputs=model.input, outputs=img_features_layer.output)
    img1_features = np.squeeze(
        features_getter.predict(np.expand_dims(img1 / 255.0, 0)))
    img2_features = np.squeeze(
        features_getter.predict(np.expand_dims(img2 / 255.0, 0)))

    return euclidian_distance(img1_features, img2_features)


def get_img_attributes_and_id(img, model):
    prediction = model.predict(np.expand_dims(img / 255.0, 0))
    attributes = np.squeeze(np.round(prediction[0]))
    img_id = np.argmax(np.squeeze(prediction[1])) + 1

    transformed_attributes = np.concatenate(
        (np.argmax(attributes[0:4]), list(attributes[4:] + 1)), axis=None)
    attributes_dict = {}
    for att_name, att_val in zip(data.get_attributes_names(), transformed_attributes):
        attributes_dict[att_name] = att_val
    return attributes_dict, img_id


def main(training=True):
    imdir = 'Market-1501'
    img1_name = '0001_c1s1_001051_00.jpg'
    img2_name = '0001_c1s1_002401_00.jpg'
    img3_name = '0992_c3s2_127469_00.jpg'

    if(training):
        model = train.train()

    model = load_model('model.h5')
    img1 = cv2.imread(os.path.join(imdir, img1_name))
    img2 = cv2.imread(os.path.join(imdir, img2_name))
    img3 = cv2.imread(os.path.join(imdir, img3_name))

    print(img_features_comparison(img1, img2, model))
    print(img_features_comparison(img1, img3, model))

    img_att, img_id = get_img_attributes_and_id(img1, model)
    print(img1_name)
    print(img_att)
    print(img_id)


if __name__ == '__main__':
    main(training=True)
