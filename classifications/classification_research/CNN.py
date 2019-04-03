import cv2
import numpy as np
import os


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class CNN:

    def __init__(self):
        self.IMG_SIZE = 50
        self.LR = 1e-3
        self.MODEL_NAME = 'dog_vs_cat-{}-{}.model'.format(self.LR, '8conv-10epoche')
        self.CURRENT = "{}/classifications/classification_research/".format(os.getcwd())

        # ----------- MODEL CREATION ----------
        convnet = input_data(shape=[None, self.IMG_SIZE, self.IMG_SIZE, 1], name='input')
        # layer
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        # layer
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        # layer
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        # layer
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        # layer
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        # layer
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        # layer
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        # layer
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)



        # layer
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        # layer
        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=self.LR, loss='categorical_crossentropy', name='targets')

        self.model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(self.CURRENT + self.MODEL_NAME)):
            self.model.load(self.CURRENT + self.MODEL_NAME)
            print('EXISTING MODEL LOADED!!!')
        else:
            print('MODEL NOT EXIST!!!')




    def prediction(self, image):

        # ----------- IMAGE CONVERION ----------
        img_np = np.fromstring(image.read(), np.uint8)
        img_decoded = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img_decoded, (self.IMG_SIZE, self.IMG_SIZE))
        img_stream = np.array(img)
        # ----------- PREDICTION ----------
        # cat = [1,0]
        # dog = [0,1]
        img_stream_preprosessed = img_stream.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
        prediction = self.model.predict([img_stream_preprosessed])[0]

        is_cat = prediction[0]
        is_dog = prediction[1]

        result = {
            "class": "UNDEFINED",
            "accuracy": 0,
        }

        if(is_cat > is_dog):
            result['class'] = 'CAT'
            result['accuracy'] = is_cat
        else:
            result['class'] = 'DOG'
            result['accuracy'] = is_dog

        return result

