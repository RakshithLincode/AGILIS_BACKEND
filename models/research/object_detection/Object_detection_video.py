######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from more_itertools import locate

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from scipy.spatial import distance as dist

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
#VIDEO_NAME = '/home/lincode/Ashirwad model/models/research/object_detection/Good.mp4'
accuracy_threshold = 0.20

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
#PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture('G:/Anode/models/research/object_detection/a.mp4')

i = 0 

out = cv2.VideoWriter('out_surfscratch.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame = cv2.resize(frame,(640,480))
    frame_expanded = np.expand_dims(frame, axis=0)
    image =frame

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    objects = []
    accuracy = []
    coordinates = []

    for index, value in enumerate(classes[0]):

      if scores[0, index] > accuracy_threshold:

        objects.append((category_index.get(value)).get('name'))
        accuracy.append(scores[0, index])

    true_boxes = boxes[0][scores[0] > accuracy_threshold]

    height, width, ch = image.shape
    for i in range(true_boxes.shape[0]):
        ymin = int(true_boxes[i,0]*height)
        xmin = int(true_boxes[i,1]*width)
        ymax = int(true_boxes[i,2]*height)
        xmax = int(true_boxes[i,3]*width)

        arr = [xmin, ymin, xmax, ymax]
        coordinates.append(arr)
     

    all_proper_u1 = list(locate(objects, lambda x: x == 'e1'))
    all_proper_u2 = list(locate(objects, lambda x: x == 's1'))
    dets = []
    #[xmin,1, ymin0, xmax2, ymax3]

    for i in all_proper_u1:
        coord = [coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3],0.99,0.99,1]
        dets.append(coord)
        detected_class = 'e1'

        crop = image[coordinates[i][1]:coordinates[i][3],coordinates[i][0]:coordinates[i][2]]
        cv2.imwrite("/home/lincode/Ashirwad model/models/research/object_detection/cropped_images/e1.jpg", crop)
        #crop = cv2.resize(crop,(640,480))
        rect = cv2.rectangle(crop,(coordinates[i][0], coordinates[i][1]) , ( coordinates[i][2], coordinates[i][3]),(0, 255, 100),2)
        rect_2 = cv2.imread("/home/lincode/Ashirwad model/models/research/object_detection/cropped_images/e1.jpg")
        centriod_rect2 = (((coordinates[i][0]+coordinates[i][2])/2), ((coordinates[i][1]+coordinates[i][3])/2))
        x2,y2 = centriod_rect2
        print("rect2---->",centriod_rect2)

    for i in all_proper_u2:
        coord = [coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3],0.99,0.99,1]
        dets.append(coord)
        detected_class = 's1'
        crop = image[coordinates[i][1]:coordinates[i][3],coordinates[i][0]:coordinates[i][2]]
        cv2.imwrite("/home/lincode/Ashirwad model/models/research/object_detection/cropped_images/s1.jpg", crop)
        #crop = cv2.resize(crop,(640,480))
        rect = cv2.rectangle(crop,(coordinates[i][0], coordinates[i][1]) , ( coordinates[i][2], coordinates[i][3]),(0, 255, 100),2)
        rect_1 = cv2.imread("/home/lincode/Ashirwad model/models/research/object_detection/cropped_images/s1.jpg")
        centriod_rect1 = (((coordinates[i][0]+coordinates[i][2])/2), ((coordinates[i][1]+coordinates[i][3])/2))
        x1,y1 = centriod_rect1
        print("rect1---->",centriod_rect1)

        pixelvalue = cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1,16)

        D = dist.euclidean((x1, y1),(x2, y2))
        mmPerPix = D/14.11
        print(1/mmPerPix)
        #reqDist = dist.euclidean((x1,y1),(x2,y2))
        #print(">>>>>>>",reqDist)
        print("-------------->",mmPerPix,"mm")


    frame = cv2.resize(frame,(1000,600))
        
        

    cv2.imshow('original',image)
    if cv2.waitKey(1) == ord('q'):
        break
    
# Clean up
video.release()
cv2.destroyAllWindows()

