from keras.models import Sequential
from keras.layers import Dense


def create_model(input_shape, n_classes=1501):
    model = Sequential()

    model.add(Dense(n_classes, activation='softmax', input_shape=(input_shape,)))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    create_model(542).summary()
