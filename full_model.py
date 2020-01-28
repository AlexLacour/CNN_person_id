from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation, Flatten, Concatenate, Conv2D, MaxPooling2D, Embedding, Reshape
from keras.applications.resnet import ResNet50
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import data


N_EPOCHS = 60
BATCH_SIZE = 32
TEST_SIZE = 0.3


def create_model(img_shape=(128, 64, 3), n_att=27, n_ids=1501):
    """
    INPUT
    """
    img_input = Input(shape=img_shape)
    cnn_backbone = ResNet50(include_top=False)(img_input)

    """
    FEATURES
    """
    features = Flatten()(cnn_backbone)
    features = Dense(512)(features)
    features = BatchNormalization()(features)
    features = Dropout(0.5)(features)
    features = Activation('relu')(features)

    """
    ATTRIBUTES
    """
    attributes = Dense(n_att,
                       activation='relu',
                       name='attributes_output')(features)

    """
    ID PREDICTION
    """
    ids = Concatenate()([features, attributes])
    ids = Dense(1024)(ids)
    ids = Dropout(0.4)(ids)

    ids = Dense(n_ids, activation='softmax',
                name='ids_output')(ids)

    """
    FULL MODEL
    """
    model = Model(inputs=img_input,
                  outputs=[attributes, ids])
    losses = {'attributes_output': euclidian_distance_loss,
              'ids_output': 'categorical_crossentropy'}
    losses_weights = {'attributes_output': 0.1,
                      'ids_output': 0.9}
    optimizer = Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer,
                  loss=losses,
                  loss_weights=losses_weights,
                  metrics=['accuracy'])

    return model


def euclidian_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def lr_schedule(epoch):
    lr = 0.01
    if(epoch >= 40):
        lr = 0.001
    return lr


def model_evaluation(model, imdir='Market-1501'):
    img_names = os.listdir(imdir)
    img_name = random.choice(img_names)
    img = cv2.imread(os.path.join(imdir, img_name)) / 255.0
    img = np.expand_dims(img, 0)
    prediction = model.predict(img)
    print(img_name)
    return prediction


if __name__ == '__main__':
    """
    DATA LOADING
    """
    X, y_att, y_id = data.data_for_full_model(preprocess_att=False)
    X = X / 255.0
    print('DATA LOADED\n')

    X_s, y_att_s, y_id_s = shuffle(X, y_att, y_id)
    print('DATA SHUFFLED\n')

    model = create_model(img_shape=X[0].shape,
                         n_att=len(y_att[0]),
                         n_ids=len(y_id[0]))
    model.summary()
    print('MODEL CREATED\n')

    X_train, X_test, y_att_train, y_att_test, y_id_train, y_id_test = train_test_split(
        X_s, y_att_s, y_id_s, test_size=TEST_SIZE, shuffle=True)
    print('DATA SPLIT\n')

    callbacks = [LearningRateScheduler(schedule=lr_schedule)]

    print('TRAINING STARTED\n')
    h = model.fit(X_train, {'attributes_output': y_att_train, 'ids_output': y_id_train},
                  validation_data=(X_test,
                                   {'attributes_output': y_att_test, 'ids_output': y_id_test}),
                  epochs=N_EPOCHS,
                  batch_size=BATCH_SIZE,
                  callbacks=callbacks,
                  shuffle=True)

    """
    SAVE MODEL
    """
    model.save_weights('full_model.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    """
    TEST MODEL
    """
    model.load_weights('full_model.h5')
    prediction = model_evaluation(model)
    print(prediction[0])
    print(np.argmax(prediction[1]))

    """
    PLOT RESULTS
    """
    loss_names = ['loss', 'attributes_output_loss', 'ids_output_loss']
    accuracy_names = ['attributes_output_accuracy', 'ids_output_accuracy']

    plt.figure()
    for i, loss in enumerate(loss_names):
        plt.subplot(3, 1, i + 1)
        plt.plot(h.history[loss])
        plt.plot(h.history['val_' + loss])
        plt.ylabel(loss)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.figure()
    for i, acc in enumerate(accuracy_names):
        plt.subplot(2, 1, i + 1)
        plt.plot(h.history[acc])
        plt.plot(h.history['val_' + acc])
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
    plt.show()
