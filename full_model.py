from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation, Flatten, Concatenate
from keras.applications.resnet import ResNet50
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import data


def create_model(img_shape=(64, 128, 3), n_att=30, n_ids=1501):
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
    attributes = Dense(n_att, activation='sigmoid',
                       name='attributes_output')(features)

    """
    ID PREDICTION
    """
    ids = Concatenate(axis=1)([features, attributes])
    ids = Dense(n_ids, activation='softmax',
                name='ids_output')(ids)

    """
    FULL MODEL
    """
    model = Model(inputs=img_input,
                  outputs=ids)

    losses = {'attributes_output': 'binary_crossentropy',
              'ids_output': 'categorical_crossentropy'}
    optimizer = Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer, loss=losses, metrics=['accuracy'])

    return model


def lr_schedule(epoch):
    lr = 0.01
    if(epoch >= 40):
        lr = 0.001
    return lr


if __name__ == '__main__':
    model = create_model()
    X, y_att, y_id = data.data_for_full_model()

    X_train, X_test, y_att_train, y_att_test, y_id_train, y_id_test = train_test_split(
        X, y_att, y_id, test_size=0.2)

    callbacks = [LearningRateScheduler(schedule=lr_schedule)]

    h = model.fit(X_train, {'attributes_output': y_att_train, 'ids_output': y_id_train},
                  validation_data=(
                      X_test, {'attributes_output': y_att_test, 'ids_output': y_id_test}),
                  epochs=60,
                  batch_size=32,
                  callbacks=callbacks)

    model.save_weights('full_model.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    loss_names = ['loss', 'attributes_output_loss', 'ids_output_loss']
    accuracy_names = ['attributes_output_acc', 'ids_output_acc']

    plt.figure()
    for i, loss in enumerate(loss_names):
        plt.subplot(3, 1, i + 1)
        plt.plot(h.history[loss])
        plt.plot(h.history['val_' + loss])
        plt.ylabel('loss')
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
