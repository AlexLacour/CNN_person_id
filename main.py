from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import data
import attributes_model
import pickle


def train_attributes_model(epochs=10, batch_size=32):
    (X_train, y_train), (X_test, y_test) = data.load_data()
    print("ATTRIBUTES DATA LOADED.")

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train,
                                         batch_size=batch_size)
    test_generator = test_datagen.flow(X_test, y_test,
                                       batch_size=batch_size)

    print("ATTRIBUTES GENERATORS CREATED.")

    input_shape = X_train.shape[1:4]
    num_attributes = y_train.shape[1]

    model = attributes_model.create_model(input_shape, num_attributes)

    checkpointer = ModelCheckpoint('attributes_model.h5',
                                   save_best_only=True,
                                   monitor='val_loss')

    print("ATTRIBUTES MODEL CREATED.")

    h = model.fit_generator(train_generator,
                            steps_per_epoch=len(X_train) // batch_size,
                            validation_data=test_generator,
                            validation_steps=len(X_test) // batch_size,
                            epochs=epochs,
                            callbacks=[checkpointer])

    print("ATTRIBUTES MODEL : TRAINING OVER.")

    with open('history_attributes_model.pickle', 'wb') as file:
        pickle.dump(h.history, file)


def main():
    train_attributes_model(epochs=5)


if __name__ == '__main__':
    main()
