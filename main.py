import train
import data
from keras.models import Model
import numpy as np
import os
import cv2
import random


def attributes_prediction_bw(att_model, imgs):
    att_prediction = att_model.predict(imgs)
    return att_prediction


def concatenate_features_attributes(features, attributes):
    id_data = []
    for feature in features:
        id_data.append(feature)
    for attribute in attributes:
        id_data.append(int(attribute))
    return np.asarray(id_data)


def generate_X_id(X_train_f, X_train_att, X_test_f, X_test_att):
    X_id = []
    for features, attributes in zip(X_train_f, X_train_att):
        id_data = concatenate_features_attributes(features, attributes)
        X_id.append(id_data)
    for features, attributes in zip(X_test_f, X_test_att):
        id_data = concatenate_features_attributes(features, attributes)
        X_id.append(id_data)
    return np.asarray(X_id)


def generate_y_id(y_tr, y_te):
    y_id = []
    for label in y_tr:
        y_id.append(label)
    for label in y_te:
        y_id.append(label)
    return np.asarray(y_id)


def final_result(img, att_model, cnn_backbone, id_model):
    img = np.expand_dims(img, 0)

    attributes = np.squeeze(att_model.predict(img))
    features = np.squeeze(cnn_backbone.predict(img))

    id_data = concatenate_features_attributes(features, attributes)
    id_data = np.expand_dims(id_data, 0)
    id_prediction = np.squeeze(id_model.predict(id_data))
    return id_prediction


def model_evaluation(att_model, cnn_backbone, id_model, imdir='Market-1501', sample_size=10):
    imgs_names = os.listdir(imdir)
    imgs_names_to_predict = random.sample(imgs_names, sample_size)
    predictions = {}
    for img_name in imgs_names_to_predict:
        img = cv2.imread(os.path.join(imdir, img_name)) / 255.0
        predictions[img_name] = final_result(
            img, att_model, cnn_backbone, id_model)
    return predictions


def main():
    """
    IMAGES + ATTRIBUTES LOADING
    """
    (X_train, y_train), (X_test, y_test) = data.load_att_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print('IMAGES + ATTRIBUTES LOADING DONE')

    """
    ATTRIBUTES MODEL
    """
    att_model = train.get_train_attributes_model(X_train, y_train,
                                                 X_test, y_test,
                                                 training=False,
                                                 train_resnet=True,
                                                 epochs=60)

    print('ATTRIBUTES MODEL DONE')

    """
    ATTRIBUTES AND FEATURES PREDICTION
    """
    cnn_backbone = Model(att_model.input, att_model.layers[2].output)

    X_train_features = cnn_backbone.predict(X_train)
    X_test_features = cnn_backbone.predict(X_test)

    X_train_att = att_model.predict(X_train)
    X_test_att = att_model.predict(X_test)

    print('ATTRIBUTES AND FEATURES PREDICTION DONE')

    """
    ATT+F => ID
    """
    y_train_id, y_test_id = data.load_ids_data()

    X_id = generate_X_id(X_train_features, X_train_att,
                         X_test_features, X_test_att)
    y_id = generate_y_id(y_train_id, y_test_id)

    print('ATT+F => ID DONE')

    """
    ID MODEL
    """
    np.save('X_id.npy', X_id)
    np.save('y_id.npy', y_id)
    id_model = train.get_train_id_model(X_id, y_id,
                                        training=False,
                                        epochs=60,
                                        val_split=0.1)

    print('ID MODEL DONE')

    """
    TEST OF ID PREDICTION
    """
    predictions = model_evaluation(att_model, cnn_backbone, id_model)
    for img_name, prediction in zip(predictions.keys(), predictions.values()):
        print(f'{img_name} => {np.argmax(prediction) + 1}')

    print('ID PREDICTION DONE')


if __name__ == '__main__':
    main()
