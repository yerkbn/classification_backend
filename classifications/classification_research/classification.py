import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

IMG_SIZE = 50
LR = 1e-3
EPOCHE = 5
MODEL_NAME = 'dog_vs_cat-{}-{}.model'.format(LR, '6conv-5epoche')
CURRENT = '/home/yerkebulan/app/dev/projects/classification_backend/classifications/classification_research/'

def prediction(image):
    # ----------- MODEL CREATION ----------
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    #layer
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    #layer
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    #layer
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    #layer
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    #layer
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    #layer
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    #layer
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    #layer
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(CURRENT+MODEL_NAME)):
        model.load(CURRENT+MODEL_NAME)
        print('EXISTING MODEL LOADED!!!')
    else:
        print('MODEL NOT EXIST!!!')

    # ----------- IMAGE CONVERION ----------

    img_np = np.fromstring(image.read(), np.uint8)
    img_decoded = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img_decoded, (IMG_SIZE, IMG_SIZE))
    img_stream = np.array(img)
    # ----------- PREDICTION ----------
    # cat = [1,0]
    # dog = [0,1]
    img_stream_preprosessed = img_stream.reshape(IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict([img_stream_preprosessed])[0]

    is_cat = prediction[0]
    is_dog = prediction[1]

    result = {
        "class": "UNDEFINED",
        "accuracy": 0,
        "name":  str(image)
    }

    if(is_cat>is_dog):
        result['class'] = 'CAT'
        result['accuracy'] = is_cat
    else:
        result['class'] = 'DOG'
        result['accuracy'] = is_dog

    return result

