from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation, Flatten, Concatenate, Multiply
from keras.applications.resnet import ResNet50
from keras.optimizers import SGD


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
    features = Activation('relu', name='img_features')(features)

    """
    ATTRIBUTES
    """
    attributes = Dense(n_att,
                       activation='sigmoid',
                       name='attributes_output')(features)

    """
    REWEIGHTING
    """
    attributes_r = Dense(n_att, activation='sigmoid')(attributes)
    attributes_r = Multiply()([attributes, attributes_r])

    """
    ID PREDICTION
    """
    ids = Concatenate()([features, attributes_r])
    ids = Dense(1024)(attributes)
    ids = BatchNormalization()(ids)
    ids = Dropout(0.5)(ids)
    ids = Activation('relu')(ids)

    ids = Dense(n_ids, activation='softmax',
                name='ids_output')(ids)

    """
    FULL MODEL
    """
    model = Model(inputs=img_input,
                  outputs=[attributes, ids])
    losses = {'attributes_output': 'binary_crossentropy',
              'ids_output': 'categorical_crossentropy'}
    losses_weights = {'attributes_output': 0.1,
                      'ids_output': 0.9}
    optimizer = SGD(learning_rate=0.01)

    model.compile(optimizer=optimizer,
                  loss=losses,
                  loss_weights=losses_weights,
                  metrics=['accuracy'])

    return model


def lr_schedule(epoch):
    lr = 0.01
    if(epoch >= 40):
        lr = 0.001
    return lr
