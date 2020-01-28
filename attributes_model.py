from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from keras.applications.resnet import ResNet50
import keras.backend as K


def euclidian_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def create_model(input_shape, n_attributes=27, train_resnet=False):
    model = Sequential()

    resnet = ResNet50(include_top=False,
                      input_shape=input_shape)

    if(not train_resnet):
        for layer in resnet.layers:
            layer.trainable = False
    model.add(resnet)
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(n_attributes, activation='relu'))

    model.compile(optimizer='adam',
                  loss=euclidian_distance_loss,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    create_model((64, 128, 3)).summary()
