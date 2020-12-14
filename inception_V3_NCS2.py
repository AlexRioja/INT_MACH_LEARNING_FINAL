from openvino.inference_engine import IENetwork, IEPlugin
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils
import cv2 
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
from utils import resize_image


def preprocess(frame):
    # convert the image pixels to a numpy array
    image = img_to_array(cv2.resize(frame,(224,224)))
    #image=cv2.resize(frame,(244,244))
    # reshape data for the model (samples, rows, columns, and channels.)
    image = image.reshape((1, image.shape[2], image.shape[1], image.shape[0]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    #image=image.astype(np.float32)
    return image

def load_ncs_model():
    """
    net = ie.load_network(ie.read_network('net.xml', 'net.bin'),"MYRIAD")
    ie.set_config({'VPU_HW_STAGES_OPTIMIZATION': 'NO'}, "MYRIAD")
    """
    model_xml="IR_model/IR_model.xml"
    model_bin="IR_model/IR_model.bin"

    plugin=IEPlugin(device="MYRIAD")
    #plugin.set_config({'VPU_HW_STAGES_OPTIMIZATION': 'NO'})
    net=IENetwork(model=model_xml,weights=model_bin)
    
    

    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))

    # Load the network and get the network shape information
    exec_net = plugin.load(network = net)
    input_shape = net.inputs[input_blob].shape
    output_shape = net.outputs[output_blob].shape



    return exec_net,input_blob,input_shape

def process_output(res,image):
    result= resize_image(image)
    pred = imagenet_utils.decode_predictions(res['dense_1/Softmax'])
    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    #print(P[0][0])
    first_pred=pred[0][0]
    #print(first_pred[1],first_pred[2])
    
    for (i, (imagenetID, label, prob)) in enumerate(pred[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
    
    
    
    # print the classification
    print('%s (%.2f%%)' % (first_pred[1], first_pred[2]*100))
    cv2.putText(result, "Prediction:"+str(first_pred[1])+" with: {:.2f}".format(first_pred[2]*100)+'%', 
                    (5, result.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                cv2.LINE_AA)
    
    cv2.imshow("Reconocimiento Inception V3", result)

net,i_b,o_s=load_ncs_model()

res=input("\nLive feed? (Y/n)")
if res=='n':
    img=cv2.imread('wine.png')
    image=preprocess(img)
    res=net.infer(inputs={"inception_v3_input":image})
    process_output(res,img)
    key = cv2.waitKey(10000) & 0xFF
else:
    fps = FPS().start()
    video_interface = cv2.VideoCapture(0)
    while True:
        ret, frame = video_interface.read()
        if not ret:
            break
        image=preprocess(frame)
        #req_handle = net.start_async(request_id=0, inputs={i_b:image})
        res=net.infer(inputs={"inception_v3_input":image})
        #print(type(res['dense_1/Softmax'][0]))
        process_output(res,frame)
        
        fps.update()
        key = cv2.waitKey(1) & 0xFF

        # rompemos bucle con la tecla q
        if key == ord("q"):
            break
    fps.stop()
    print("FPS aproximados: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()



