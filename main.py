import train
import data
from keras.models import Model
import numpy as np
import os
import cv2


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


def main():
    """
    IMAGES + ATTRIBUTES LOADING
    """
    (X_train, y_train), (X_test, y_test) = data.load_att_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    """
    ATTRIBUTES MODEL
    """
    att_model = train.get_train_attributes_model(X_train, y_train,
                                                 X_test, y_test,
                                                 training=False,
                                                 train_resnet=True,
                                                 epochs=60)

    """
    ATTRIBUTES AND FEATURES PREDICTION
    """
    cnn_backbone = Model(att_model.input, att_model.layers[2].output)

    X_train_features = cnn_backbone.predict(X_train)
    X_test_features = cnn_backbone.predict(X_test)

    X_train_att = att_model.predict(X_train)
    X_test_att = att_model.predict(X_test)

    """
    ATT+F => ID
    """
    y_train_id, y_test_id = data.load_ids_data()

    X_id = generate_X_id(X_train_features, X_train_att,
                         X_test_features, X_test_att)
    y_id = generate_y_id(y_train_id, y_test_id)

    """
    ID MODEL
    """
    id_model = train.get_train_id_model(X_id, y_id,
                                        training=True,
                                        epochs=100,
                                        val_split=0.3)

    """
    TEST OF ID PREDICTION
    """
    img = cv2.imread('Market-1501/0001_c1s1_001051_00.jpg')
    predicted_class = final_result(img, att_model, cnn_backbone, id_model)
    print(predicted_class)
    print(np.max(predicted_class))
    print(f'Predicted class : {np.argmax(predicted_class)}')


if __name__ == '__main__':
    main()
