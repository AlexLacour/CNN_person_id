from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import data
import attributes_model
import pickle


def main():
    (X_train, y_train), (X_test, y_test) = data.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    input_shape = X_train.shape[1:4]
    num_attributes = y_train.shape[1]
    batch_size = 32

    checkpointer = ModelCheckpoint('cnn_re_id_weights.h5',
                                   save_best_only=True,
                                   monitor='val_loss')

    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train,
                                         batch_size=batch_size)
    test_generator = test_datagen.flow(X_test, y_test,
                                       batch_size=batch_size)

    model = attributes_model.create_model(input_shape, num_attributes)

    h = model.fit_generator(train_generator,
                            steps_per_epoch=len(X_train) // batch_size,
                            validation_data=test_generator,
                            validation_steps=len(X_test) // batch_size,
                            epochs=100,
                            callbacks=[checkpointer])

    with open('history_cnn_re_id.pickle', 'wb') as file:
        pickle.dump(h.history, file)


if __name__ == '__main__':
    main()
