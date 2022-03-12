# How to load a Tensorflow model using OpenCV
# Jean Vitor de Paulo Blog - https://jeanvitor.com/tensorflow-object-detecion-opencv/
 
import cv2
import numpy as np
# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('D:/deploy/Model-2-22/frozen_inference_graph.pb', 'D:/deploy/Model-2-22/graph.pbtxt')

cv2.IMREAD_UNCHANGED
# Input image
img = cv2.imread("D:\\check\\dianbaoloushi-result\\rotate\\210CQK10069_2_topfront1.jpg",cv2.IMREAD_COLOR)
rows, cols, channels = img.shape
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("D:\\check\\dianbaoloushi-result\\rotate\\rImg.jpg", img)
cv2.waitKey(0)
# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(500,500), swapRB=True, crop=False))
# Runs a forward pass to compute the net output
networkOutput = tensorflowNet.forward()

# Loop on the outputs
for detection in networkOutput[0,0]:
    
    score = float(detection[2])
    if score > 0.5:
        print(detection)
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
 
        #draw a red rectangle around detected objects
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=1)
 
# Show the image with a rectagle surrounding the detected objects 
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()