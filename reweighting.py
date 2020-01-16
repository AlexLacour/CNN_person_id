from keras.models import Sequential
from keras.layers import Dense


def attributes_reweighting(n_attributes):
    reweighting_model = Sequential()
    reweighting_model.add(
        Dense(n_attributes, input_shape=(n_attributes,), activation='sigmoid'))
    reweighting_model.compile(optimizer='adam', loss='mse')

    return reweighting_model
