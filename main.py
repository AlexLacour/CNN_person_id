import cv2
import numpy as np
import train


def attributes_prediction(att_model, img):
    img = np.expand_dims(img, 0)
    att_prediction = att_model.predict(img)
    att_prediction = np.squeeze(att_prediction)
    att_prediction[0] *= 3
    att_prediction = att_prediction.astype(int)
    return att_prediction


def main():
    att_model = train.get_train_attributes_model()

    img = cv2.imread('Market-1501/0001_c1s1_002301_00.jpg')
    att_prediction = attributes_prediction(att_model, img)
    print(att_prediction)


if __name__ == '__main__':
    main()
