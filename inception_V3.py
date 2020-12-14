from keras.applications import InceptionV3
from keras.preprocessing.image import load_img
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import cv2 
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import tensorflow as tf
import os
import keras
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
from utils import resize_image
#keras.backend.clear_session()
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from keras.models import model_from_json


def preprocess_and_inference(frame,pretrained):
    result= resize_image(frame)
    # convert the image pixels to a numpy array
    if pretrained:
        image = img_to_array(cv2.resize(frame,(299,299)))
    else:
        image = img_to_array(cv2.resize(frame,(224,224)))
    # reshape data for the model (samples, rows, columns, and channels.)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    pred = model.predict(image)
    pred = imagenet_utils.decode_predictions(pred)
    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    #print(P[0][0])
    first_pred=pred[0][0]
    print(first_pred[1],first_pred[2])
    '''
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
    '''
    
    # print the classification
    print('%s (%.2f%%)' % (first_pred[1], first_pred[2]*100))
    cv2.putText(result, "Prediction:"+str(first_pred[1])+" with: {:.2f}".format(first_pred[2]*100)+'%', 
                    (5, result.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                cv2.LINE_AA)
    
    cv2.imshow("Reconocimiento Inception V3", result)
    
res=input("Use pretrained or not?(y/N):")
pretrained=False
if res=='y':
    pretrained=True
    #____________________FIRST OPTION, PRETRAINED 
    #   
    model = InceptionV3(weights="imagenet")
else:
    #____________________SECOND OPTION, TRANSFER LEARNING
    # load json and create model
    json_file = open('Trained_by_me_models/InceptionV3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("Trained_by_me_models/InceptionV3.h5")
    print("Loaded model from disk")

res2=input("\nLive feed? (Y/n)")
if res2=='n':
    img=cv2.imread('cat.png')
    preprocess_and_inference(img,pretrained)
    key = cv2.waitKey(10000) & 0xFF
else:
    # contador aproximado de FPS
    fps = FPS().start()
    video_interface = cv2.VideoCapture(0)

    while True:
        ret, frame = video_interface.read()
        if not ret:
            break
        preprocess_and_inference(frame,pretrained)
        fps.update()
        key = cv2.waitKey(1) & 0xFF

        # rompemos bucle con la tecla q
        if key == ord("q"):
            break
    fps.stop()
    print("FPS aproximados: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()

