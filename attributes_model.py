from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from keras.applications.resnet import ResNet50


def create_model(input_shape, n_attributes=27):
    model = Sequential()
    model.add(ResNet50(include_top=False, input_shape=input_shape))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(n_attributes, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    create_model((64, 128, 3)).summary()
