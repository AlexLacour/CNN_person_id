from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout


def create_model(input_dim, n_classes=1501):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    create_model(539).summary()
