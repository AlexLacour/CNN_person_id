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


def generate_X_id(X_features, X_attributes):
    X_id = []
    for features, attributes in zip(X_features, X_attributes):
        id_data = concatenate_features_attributes(features, attributes)
        X_id.append(id_data)
    return np.asarray(X_id)


def final_result(img, att_model, cnn_backbone, id_model):
    img = np.expand_dims(img, 0)

    attributes = np.squeeze(att_model.predict(img))
    features = np.squeeze(cnn_backbone.predict(img))
    id_data = concatenate_features_attributes(features, attributes)
    id_data = np.expand_dims(id_data, 0)
    id_prediction = np.squeeze(id_model.predict(id_data))
    return id_prediction


def main():
    (X_train, y_train), (X_test, y_test) = data.load_att_data()

    att_model = train.get_train_attributes_model(X_train, y_train,
                                                 X_test, y_test,
                                                 training=False,
                                                 train_resnet=True,
                                                 epochs=60)

    cnn_backbone = Model(att_model.input, att_model.layers[2].output)

    X_train_features = cnn_backbone.predict(X_train)
    X_test_features = cnn_backbone.predict(X_test)

    X_train_att = att_model.predict(X_train)
    X_test_att = att_model.predict(X_test)

    X_train_id = generate_X_id(X_train_features, X_train_att)
    X_test_id = generate_X_id(X_test_features, X_test_att)
    y_train_id, y_test_id = data.load_ids_data()

    print(X_train_id.shape)
    print(y_train_id.shape)

    id_model = train.get_train_id_model(X_train_id, y_train_id,
                                        X_test_id, y_test_id,
                                        training=True)

    img_names = os.listdir('Market-1501')
    img_name = np.random.choice(img_names)
    img = cv2.imread(img_name)

    id_of_img = final_result(img, att_model, cnn_backbone, id_model)
    print(f'Final result = {np.argmax(id_of_img)} : {np.max(id_of_img)}')


if __name__ == '__main__':
    main()
