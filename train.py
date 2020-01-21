from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import attributes_model
import history_plot as h_plt
import pickle
import reweighting
import id_model

"""
SEPARATE TRAINING FUNCTIONS
"""


def get_train_attributes_model(X_train, y_train, X_test, y_test, epochs=60, batch_size=32, training=True, train_resnet=False):
  train_datagen = ImageDataGenerator()
  test_datagen = ImageDataGenerator()

  train_generator = train_datagen.flow(X_train, y_train,
                                       batch_size=batch_size)
  test_generator = test_datagen.flow(X_test, y_test,
                                     batch_size=batch_size)

  input_shape = X_train.shape[1:4]
  num_attributes = y_train.shape[1]

  model = attributes_model.create_model(
      input_shape, num_attributes, train_resnet)

  if(training):
    checkpointer = ModelCheckpoint('attributes_model.h5',
                                   save_best_only=True,
                                   monitor='val_loss')

    h = model.fit_generator(train_generator,
                            steps_per_epoch=len(X_train) // batch_size,
                            validation_data=test_generator,
                            validation_steps=len(X_test) // batch_size,
                            epochs=epochs,
                            callbacks=[checkpointer])

    with open('history_attributes_model.pickle', 'wb') as file:
      pickle.dump(h.history, file)
    h_plt.plot_loss_acc(h)

  model = load_model('attributes_model.h5')

  return model


def get_train_reweight_model(pred_train, y_train, pred_test, y_test, epochs=60, batch_size=32, training=True):
  model = reweighting.create_model(len(pred_train[0]))
  if(training):
    checkpointer = ModelCheckpoint('reweighting_model.h5',
                                   save_best_only=True,
                                   monitor='val_loss')
    h = model.fit(pred_train, y_train, epochs=epochs,
                  batch_size=batch_size, callbacks=[checkpointer], validation_data=(pred_test, y_test))
    h_plt.plot_loss_acc(h)
  model = load_model('reweighting_model.h5')

  return model


def get_train_id_model(X, y, epochs=60, batch_size=32, training=True, val_split=0.3):
  model = id_model.create_model(len(X[0]))
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=val_split)
  if(training):
    checkpointer = ModelCheckpoint('id_model.h5',
                                   save_best_only=True,
                                   monitor='val_loss')
    h = model.fit(X_train, y_train, epochs=epochs,
                  batch_size=batch_size, callbacks=[checkpointer], validation_data=(X_test, y_test))
    h_plt.plot_loss_acc(h)
  model = load_model('id_model.h5')

  return model
