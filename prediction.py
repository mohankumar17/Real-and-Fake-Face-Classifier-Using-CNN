import tensorflow
import keras
import numpy as np
from keras.models import load_model
import cv2
cnn_model=load_model('real_fake_face_model_2.h5')
cnn_model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

from skimage.transform import resize

def detect(frame):
    try:
        img=resize(frame,(128,128))
        img=np.expand_dims(img,axis=0)

        if(np.max(img)>1):
            img=img/255.0

        prediction=cnn_model.predict(img)
        prediction=cnn_model.predict_classes(img)
        print(prediction)

        if prediction>=0.5:
            classp='Real'
        else:
            classp='Fake'
        print(classp)

    except AttributeError:
        print('Shape not found')

#frame=cv2.imread(r"D:\AAA\COURSES\AI\PROJECT\malignant_test_image.jpg")

frame=cv2.imread(r"testset\fake\easy_31_0011.jpg")
data=detect(frame)

frame=cv2.imread(r"testset\fake\easy_32_1100.jpg")
data=detect(frame)

frame=cv2.imread(r"testset\fake\easy_33_0010.jpg")
data=detect(frame)

frame=cv2.imread(r"testset\real\real_00005.jpg")
data=detect(frame)

frame=cv2.imread(r"testset\real\real_00007.jpg")
data=detect(frame)

frame=cv2.imread(r"testset\real\real_00009.jpg")
data=detect(frame)
