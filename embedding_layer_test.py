from keras.models import Sequential
from keras.layers import Embedding
import numpy as np
import random

input_data = [1 if random.random() < 0.5 else 2 for _ in range(27)]
input_data = np.expand_dims(input_data, 0)
print(input_data)

model = Sequential()
model.add(Embedding(input_dim=np.max(input_data),
                    output_dim=1, input_length=27))

prediction = model.predict(input_data)
print(prediction.shape)
print(prediction)
