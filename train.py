from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import data
import reid_model


N_EPOCHS = 60
BATCH_SIZE = 32
TEST_SIZE = 0.2
PREPRO_ATT = True


"""
AUGMENTATION
"""


def random_cropping(img):
    crop_pctg = 0.75
    cropping_chance = 0.5
    if(random.random() < cropping_chance):
        img_n_rows = img.shape[0]
        img_n_cols = img.shape[1]
        crop_img_n_rows = int(crop_pctg * img_n_rows)
        crop_img_n_cols = int(crop_pctg * img_n_cols)
        row_start = random.randint(0, (1 - crop_pctg) * img_n_rows)
        col_start = random.randint(0, (1 - crop_pctg) * img_n_cols)

        cropped_img = img[row_start:crop_img_n_rows,
                          col_start:crop_img_n_cols,
                          :]
        cropped_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]))
    else:
        cropped_img = img[:, :, :]
    return cropped_img


def data_generator(X, y_att, y_id, gen):
    gen_att = gen.flow(X, y_att, batch_size=BATCH_SIZE, seed=42, shuffle=False)
    gen_id = gen.flow(X, y_id, batch_size=BATCH_SIZE, seed=42, shuffle=False)

    while True:
        att = gen_att.next()
        ids = gen_id.next()
        yield att[0], [att[1], ids[1]]


"""
VISUALIZATION
"""


def plot_history(h):
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


"""
METRICS
"""


def model_sample_test(model, imdir='Market-1501'):
    img_names = os.listdir(imdir)
    img_name = random.choice(img_names)
    img = cv2.imread(os.path.join(imdir, img_name)) / 255.0
    img = np.expand_dims(img, 0)
    prediction = model.predict(img)
    print(img_name)
    print(np.round(prediction[0]))
    print(np.argmax(prediction[1]) + 1)


def mAP(y, predicted_y):
    y_classes = [np.argmax(element) for element in y]
    predicted_y_classes = [
        np.argmax(element) for element in predicted_y]

    n_occurences_classes = np.zeros((1501))
    for class_value in y_classes:
        n_occurences_classes[class_value] += 1

    positive_classes = np.zeros((1501))
    for y_true, y_pred in zip(y_classes, predicted_y_classes):
        if(y_true == y_pred):
            positive_classes[y_true] += 1

    print(n_occurences_classes)

    AP = [(t_p / total)
          for t_p, total in zip(positive_classes, n_occurences_classes)]
    AP = np.asarray(AP)[np.logical_not(np.isnan(AP))]
    return np.mean(AP)


def rank_accuracy(y, predicted_y_prob, rank=1):
    yn_classes = [np.argmax(element) for element in y]
    predicted_y_top_classes = [
        np.argsort(element)[-rank:] for element in predicted_y_prob]
    rank_accuracy = 0
    for y_true, y_pred_prob in zip(yn_classes, predicted_y_top_classes):
        if(y_true in y_pred_prob):
            rank_accuracy += 1
    return rank_accuracy / len(y)


def model_metrics(model, X_train, y_train, X_test, y_test):
    """
    METRICS ON TRAIN SET
    """
    predicted_y_train = model.predict(X_train)[:][1]

    metrics_train = {}
    metrics_train['mAP'] = mAP(y_train, predicted_y_train)
    metrics_train['rank_1'] = rank_accuracy(y_train, predicted_y_train, 1)
    metrics_train['rank_5'] = rank_accuracy(y_train, predicted_y_train, 5)
    metrics_train['rank_10'] = rank_accuracy(y_train, predicted_y_train, 10)

    """
    METRICS ON TEST SET
    """
    predicted_y_test = model.predict(X_test)[:][1]

    metrics_test = {}
    metrics_test['mAP'] = mAP(y_test, predicted_y_test)
    metrics_test['rank_1'] = rank_accuracy(y_test, predicted_y_test, 1)
    metrics_test['rank_5'] = rank_accuracy(y_test, predicted_y_test, 5)
    metrics_test['rank_10'] = rank_accuracy(y_test, predicted_y_test, 10)

    metrics = {}
    metrics['train'] = metrics_train
    metrics['test'] = metrics_test

    return metrics


def train():
    """
    DATA LOADING
    """
    X, y_att, y_id = data.data_for_full_model(preprocess_att=PREPRO_ATT)
    print('DATA LOADED\n')

    """
    MODEL CREATION
    """
    model = reid_model.create_model(img_shape=X[0].shape,
                                    n_att=len(y_att[0]),
                                    n_ids=len(y_id[0]))
    model.summary()
    callbacks = [LearningRateScheduler(schedule=reid_model.lr_schedule)]
    print('MODEL CREATED\n')

    """
    DATA AUGMENTATION
    """
    X_train, X_test, y_att_train, y_att_test, y_id_train, y_id_test = train_test_split(
        X, y_att, y_id, test_size=TEST_SIZE, shuffle=True)
    print('DATA SPLIT\n')

    generator = ImageDataGenerator(rescale=1. / 255.,
                                   horizontal_flip=True,
                                   preprocessing_function=random_cropping)

    train_generator = data_generator(
        X_train, y_att_train, y_id_train, generator)
    test_generator = data_generator(X_test, y_att_test, y_id_test, generator)

    """
    MODEL TRAINING
    """
    print('TRAINING STARTED\n')
    h = model.fit_generator(train_generator,
                            epochs=N_EPOCHS,
                            steps_per_epoch=len(X_train) // BATCH_SIZE,
                            validation_data=test_generator,
                            validation_steps=len(X_test) // BATCH_SIZE,
                            verbose=2,
                            callbacks=callbacks)

    """
    SAVE MODEL
    """
    model.save('model.h5')

    """
    PLOT RESULTS
    """
    plot_history(h)

    """
    METRICS
    """
    print(model_metrics(model, X_train / 255.0,
                        y_id_train, X_test / 255.0, y_id_test))

    return model


if __name__ == '__main__':
    train()
