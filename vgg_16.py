from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import img_to_array
import cv2 
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import tensorflow as tf
import os
import keras
#keras.backend.clear_session()
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def preprocess_and_inference(frame):
    # convert the image pixels to a numpy array
    image = img_to_array(cv2.resize(frame,(224,224)))
    # reshape data for the model (samples, rows, columns, and channels.)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    pred = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(pred)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))
    cv2.putText(frame, "Prediction:"+str(label[1])+" with: {:.2f}".format(label[2]*100)+'%',(5,frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN,1.3,(66,53, 243), 2,cv2.LINE_AA)
    cv2.imshow("Reconocimiento VGG16", frame)

    


model = VGG16()


# contador aproximado de FPS
fps = FPS().start()
video_interface = cv2.VideoCapture(0)



while True:
    ret, frame = video_interface.read()
    if not ret:
        break
    preprocess_and_inference(frame)
    fps.update()
    key = cv2.waitKey(1) & 0xFF

    # rompemos bucle con la tecla q
    if key == ord("q"):
        break
fps.stop()
print("FPS aproximados: {:.2f}".format(fps.fps()))



