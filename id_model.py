from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def create_model(input_shape=542, n_classes=1501, n_hidden=1, n_neuron=1024, multiplier=2, activation='relu', first_activation='relu', norm=True, dropout_rate=0.5):
    model = Sequential()

    model.add(Dense(n_neuron,
                    activation=first_activation,
                    input_shape=(input_shape,)))
    if(norm):
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for _ in range(n_hidden):
        model.add(Dense(n_neuron * multiplier,
                        activation=activation))
        if(norm):
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    X = np.load('X_id.npy')
    y = np.load('y_id.npy')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_hidden = [0, 1, 2, 3]
    n_neuron = [512, 1024]
    multiplier = [1, 2]
    activation = ['relu']
    first_activation = ['relu']
    norm = [True, False]
    dropout_rate = [0, 0.2, 0.4, 0.5]
    epochs = [50]
    batch_sizes = [32]

    skmodel = KerasClassifier(build_fn=create_model)
    param_random_search = dict(epochs=epochs, batch_size=batch_sizes, n_hidden=n_hidden, n_neuron=n_neuron, multiplier=multiplier,
                               activation=activation, first_activation=first_activation, norm=norm, dropout_rate=dropout_rate)
    random_search = RandomizedSearchCV(
        estimator=skmodel, param_distributions=param_random_search)
    random_search_results = random_search.fit(X_train, y_train)

    print(
        f'Best : {random_search_results.best_score_*100}% using {random_search_results.best_params_}')
    """
    Best : 72.07403550545867% using
    {'norm': True, 'n_neuron': 1024, 'n_hidden': 0, 'multiplier': 2, 'first_activation': 'relu', 'epochs': 50, 'dropout_rate': 0.4, 'batch_size': 32, 'activation': 'relu'}
    """
