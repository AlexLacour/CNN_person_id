import matplotlib.pyplot as plt
import pickle


def plot_attributes_model():
    file = open('history_attributes_model.pickle', 'rb')
    history = pickle.load(file)
    file.close()

    print(history.keys())

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()


if __name__ == '__main__':
    plot_attributes_model()
