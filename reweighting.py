from keras.models import Sequential
from keras.layers import Dense


def create_model(n_attributes=30):
    reweighting_model = Sequential()
    reweighting_model.add(
        Dense(n_attributes, input_shape=(n_attributes,), activation='sigmoid'))
    reweighting_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return reweighting_model
