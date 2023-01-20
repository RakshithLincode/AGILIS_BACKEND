import tensorflow as tf
import cv2
import os
import sys
from livis.models.research.object_detection.utils import label_map_util
import numpy as np


def load_gvx_model():

    accuracy_threshold = 0.90
    gpu_fraction = 0.4


    NUM_CLASSES = 90
    CWD_PATH = os.getcwd()


    PATH_TO_CKPT = "/critical_data/trained_models/GVM/frozen_inference_graph.pb"
    PATH_TO_LABELS = "/critical_data/trained_models/GVM/labelmap.pbtxt"
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()

    sess = None

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,allow_growth=True)
        sess = tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options))
        print("\n\n\n MODEL LOADED \n\n\n")

    return detection_graph,sess,accuracy_threshold,category_index



def detect_gvx_frame(image,detection_graph,sess,accuracy_threshold,category_index):
    #print(image)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})  

    #print(boxes)
    #print(scores)
    #print(classes)

    objects = []
    accuracy = []
    #coordinates = []

    for index, value in enumerate(classes[0]):

        if scores[0, index] > accuracy_threshold:
            objects.append((category_index.get(value)).get('name'))
            accuracy.append(scores[0, index])



    true_boxes = boxes[0][scores[0] > accuracy_threshold]

    #height, width, ch = image.shape
    #for i in range(true_boxes.shape[0]):
    #    ymin = int(true_boxes[i,0]*height)
    #    xmin = int(true_boxes[i,1]*width)
    #    ymax = int(true_boxes[i,2]*height)
    #    xmax = int(true_boxes[i,3]*width)

    #    arr = [xmin, ymin, xmax, ymax]
    #    coordinates.append(arr)

    return scores,boxes,objects,accuracy


# use this in the calling function
#scores,boxes,objects = detect_frame(image,detection_graph,sess,accuracy_threshold)
