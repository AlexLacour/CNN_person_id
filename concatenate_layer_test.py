from keras.models import Model
from keras.layers import Input, Dense, Concatenate
import numpy as np

if __name__ == '__main__':
    input_a = Input(shape=(2,))
    input_b = Input(shape=(1,))
    layer_a = Dense(16)(input_a)
    layer_b = Dense(16)(input_b)
    layer = Concatenate()([layer_a, layer_b])
    output = Dense(1)(layer)

    model = Model(inputs=[input_a, input_b], outputs=output)

    model.compile(optimizer='adam', loss='mse')

    model.summary()

    input_test = [[1, 2], 3]
    input_test = np.asarray(input_test)

    model.predict(input_test)
