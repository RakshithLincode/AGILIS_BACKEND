from flask import Flask,request,jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time


sys.path.append("..")


from utils import label_map_util


MODEL_NAME = 'inference_graph'



CWD_PATH = os.getcwd()

#frozen inf grp 
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

#labelmap
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')



NUM_CLASSES = 10000


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')




app = Flask(__name__) 


@app.route('/predict',methods = ['POST', 'GET'])
def pred(): 

    a = time.time()
    content = request.json
    image = content['instances']
   
   
  
    ####### change
    image = cv2.imread(image)
    image = cv2.resize(image,(720,1280))
    
    
    image_expanded = np.expand_dims(image, axis=0)


    (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})


    
    objects = []
    accuracy = []
    for index, value in enumerate(classes[0]):
        object_dict = {}
        if scores[0, index] > 0.70:
            object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                        scores[0, index]
            objects.append(object_dict)
            accuracy.append(scores[0, index])

    print("totaltime",str(time.time()-a))
    final_val = None
    if len(objects) == 0:
        #no predictions
        final_val = None

    elif len(objects) == 1:
        print("one predi")
        #one prediction (maybe true or maybe false prediction)
        predicted_obj = objects[0]
        l = list(predicted_obj.keys())
        predicted_obj = l[0].decode('ascii')
        
        if '_' in predicted_obj:
            predicted_obj = str(predicted_obj.split('_')[0])
        final_val = predicted_obj

    else:
        print("multi predi")
        #multiple predictions (sort acc to accuracy and take highest acc)
        original_lst = accuracy.copy()
        if len(accuracy) > 0:
            accuracy.sort(reverse = True)

        first_ele = accuracy[0]

        idx_acc = original_lst.index(first_ele)
        predicted_obj = objects[idx_acc]
        
        l = list(predicted_obj.keys())
        predicted_obj = l[0].decode('ascii')
        
        if '_' in predicted_obj:
            predicted_obj = str(predicted_obj.split('_')[0])
        final_val = predicted_obj
        
    return final_val
  

if __name__ == '__main__': 
  
    app.run(host='127.0.0.1',port=5001,debug=True) 
