#from common.utils import MongoHelper, run_in_background
from common.utils import MongoHelper
from common.utils import RedisKeyBuilderServer,CacheHelper,MongoHelper
from bson import ObjectId
from zipfile import ZipFile
import os
import json
import uuid
import cv2
import datetime
from copy import deepcopy
import xml.etree.cElementTree as ET
from lxml import etree
import csv
from livis.constants import *
import shutil
import imutils
import random
from django.conf import settings
from accounts.utils import get_user_account_util
import tensorflow as tf
#from livis.Monk_Object_Detection.tf_obj_2.lib.models.research.object_detection.webcam import load_model_to_memory,crop_infer
from livis.models.research.object_detection.utils import label_map_util

from livis.celeryy import app
from celery import shared_task
#from livis.common_model import detection_graphm,detection_graphx,sessm,sessx,accuracy_thresholdm,accuracy_thresholdx,category_indexm,category_indexx

import datetime
import base64
import numpy as np
import requests
import cv2
import time
import requests
import cv2
import time
import gc
from billiard import Process

import subprocess

from subprocess import PIPE
from subprocess import call

def get_pred_tf_serving(image,port,model_type,NUM_CLASSES,PATH_TO_LABELS):
    NUM_CLASSES = int(NUM_CLASSES)
    if model_type == "GVM":

        image_expanded = np.expand_dims(image, axis=0)
        accuracy_threshold = 0.7
        gpu_fraction = 0.4


        #NUM_CLASSES = 18
        PATH_TO_CKPT = "/critical_data/trained_models/GVM/frozen_inference_graph.pb"
        #PATH_TO_LABELS = "/home/schneider/Documents/critical_data/trained_models/GVM/labelmap.pbtxt"
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)


        total_time = time.time()
        
        data = json.dumps({ 
        "instances": image_expanded.tolist()
        })
        SERVER_URL = 'http://localhost:'+str(port)+'/v1/models/saved_model:predict'
        #predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes.decode('utf-8')
        response = requests.post(SERVER_URL, data=data)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()['predictions'][0]
        #print(prediction.keys())
        #print('Prediction class: {}, avg latency: {} ms'.format(prediction['detection_classes'], (time.time() - total_time)))
        objects = []
        #print(prediction['detection_scores'])
        accuracy = []
        #coordinates = []

        for index, value in enumerate(prediction['detection_classes']):
            if prediction['detection_scores'][index] > accuracy_threshold:
                objects.append((category_index.get(value)).get('name'))
                accuracy.append(prediction['detection_scores'][index])

        ret_obj = {
            "scores": prediction['detection_scores'],
            "boxes": prediction['detection_boxes'],
            "objects": objects,
            "accuracy": accuracy
        }
        return ret_obj

    elif model_type == "GVX":


        image_expanded = np.expand_dims(image, axis=0)
        accuracy_threshold = 0.7
        gpu_fraction = 0.4


        NUM_CLASSES = 10
        PATH_TO_CKPT = "/critical_data/trained_models/GVX/frozen_inference_graph.pb"
        PATH_TO_LABELS = "/home/schneider/Documents/critical_data/trained_models/GVX/labelmap.pbtxt"
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)


        total_time = time.time()
        
        data = json.dumps({ 
        "instances": image_expanded.tolist()
        })
        SERVER_URL = 'http://localhost:'+str(port)+'/v1/models/saved_model:predict'
        #predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes.decode('utf-8')
        response = requests.post(SERVER_URL, data=data)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()['predictions'][0]
        #print(prediction.keys())
        #print('Prediction class: {}, avg latency: {} ms'.format(prediction['detection_classes'], (time.time() - total_time)))
        objects = []
        #print(prediction['detection_scores'])
        accuracy = []
        #coordinates = []

        for index, value in enumerate(prediction['detection_classes']):
            if prediction['detection_scores'][index] > accuracy_threshold:
                objects.append((category_index.get(value)).get('name'))
                accuracy.append(prediction['detection_scores'][index])

        ret_obj = {
            "scores": prediction['detection_scores'],
            "boxes": prediction['detection_boxes'],
            "objects": objects,
            "accuracy": accuracy
        }
        return ret_obj



def start_inspection(data):


    try:
        mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
    except:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code


    p = [p for p in mp.find()]

    p=p[0]

    workstation_id = p['_id']

    feed_urls = []
    workstation_info = RedisKeyBuilderServer(workstation_id).workstation_info
    #print(workstation_info)
    cam=workstation_info['camera_config']
    #print(cam)
    for camera_info in cam['cameras']:
        url = "http://127.0.0.1:8000/livis/v1/preprocess/stream/{}/{}/".format(workstation_id,camera_info['camera_id'])

        #feed_urls[camera_info['camera_name']] = url
        feed_urls.append(url)


    jig_id = data['jig_id']
    jig_type = data['jig_type']
    barcode_id = data['barcode']

    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    jig_id =  data['jig_id']
    if jig_id is None:
        message = "jig id not provided"
        status_code = 400
        return message,status_code

    try:
        dataset = mp.find_one({'_id' : ObjectId(jig_id)})
        if dataset is None:
            message = "Jig not found in Jig collection"
            status_code = 404
            return message,status_code


    except Exception as e:
        message = "Invalid jigID"
        status_code = 400
        return message,status_code
    

    oem_number = dataset['oem_number']
    jig_type = dataset['jig_type']

    try:
        kanban = dataset['kanban']
    except:
        return "kanban not defined",None
    try:
        vendor_match = dataset['vendor_match']
    except:
        pass
    try:
        full_img = dataset['full_img']
    except:
        return "regions not defined",None
    try:   
        user_id = data['user_id']
    except:
        return "userid not defined",None

    user_details = get_user_account_util(user_id)
    #print("user_details::::",user_details)
    role_name = user_details['role_name']
    #print("role_name::::",role_name,type(role_name))
    user = { "user_id": user_id,
                "role": user_details['role_name'],
                "name": (user_details['first_name']+" "+user_details['last_name'])
            }
    #print("user:::: ",user)
    createdAt = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    mp = MongoHelper().getCollection('INSPECTION')

    #dataset['camera_url'] = feed_urls[0]
    #dataset['jig_id'] = jig_id
    #dataset['user'] = user
    #dataset['status'] = 'started'
    #dataset['createdAt'] = createdAt
    #dataset['is_manual_pass'] = False
    #dataset['is_compleated'] = False


    obj = {
        'jig_details': dataset,
        'camera_url' : feed_urls[0],
        'jig_id' : jig_id,
        'user' : user,
        "status" : 'started',
        'createdAt' : createdAt,
        'is_manual_pass' : False,
        'is_reject':False,
        'is_compleated' : False,
        'serial_no' : barcode_id,
        'is_admin_report_reset':False
    }

    _id = mp.insert(obj)

    rch = CacheHelper()

    #rch.set_json({_id:None})

    resp = obj
    if resp:
        return resp,_id
    else:
        return {}




@shared_task
def start_real_inspection(data1,inspection_id):

    frame = None
    crp = None

    #goto workstation and fetch redis keys and camera_name
    #goto jig and pull regions
    #load ssd to memory
    #combine the regions with camera_name
    #iterate through keys,camera_name
        
        #iterate through regions
            #crop send - get back the inference
    


    ############# worstation red key and cam name

    mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)

    p = [p for p in mp.find()]

    workstation_id = p[0]['_id']

    data = RedisKeyBuilderServer(workstation_id).workstation_info

    rch = CacheHelper()

    cam_list = []
    key_list = []

    for cam in data['camera_config']['cameras']:
        camera_index = cam['camera_id']
        camera_name = cam['camera_name']
        key = RedisKeyBuilderServer(workstation_id).get_key(cam['camera_id'],'original-frame') 
        cam_list.append(cam['camera_name'])
        key_list.append(key)



    ########### jig details 

    try:
        mp = MongoHelper().getCollection('WEIGHTS')
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    s = [p for p in mp.find()]

    s=s[0]

    gvm_labelmap_pth = s['gvm_labelmap_pth']
    gvx_labelmap_pth = s['gvx_labelmap_pth']
    gvm_num_classes = s['gvm_num_classes']
    gvx_num_classes = s['gvx_num_classes']
    gvm_saved_model_pth = s['gvm_saved_model_pth']
    gvx_saved_model_pth = s['gvx_saved_model_pth']

    


    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    jig_id =  data1['jig_id']
    if jig_id is None:
        message = "jig id not provided"
        status_code = 400
        return message,status_code

    try:
        dataset = mp.find_one({'_id' : ObjectId(jig_id)})
        if dataset is None:
            message = "Jig not found in Jig collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid jigID"
        status_code = 400
        return message,status_code
    

    oem_number = dataset['oem_number']
    jig_type = dataset['jig_type']
    vendor_match = dataset['vendor_match']
    print("vendor match is")
    print(vendor_match)


    try:
        kanban = dataset['kanban']
        #print("KANBAN" , kanban )
    except:
        message = "error in kanban/not set"
        status_code = 400
        return message,status_code

    if kanban is None:
        message = "error in kanban/not set"
        status_code = 400
        return message,status_code


    try:
        #var = str(ObjectId(dataset['_id'])) + "_full_img"
        #full_img = rch.get_json(var)
        #print(full_img)
        from os import path
        import json 
        full_img = None
        print("999999999999")
        print(path.exists('/critical_data/regions/'+str(ObjectId(dataset['_id'])) + "_full_img"+".json"))
        if path.exists('/critical_data/regions/'+str(ObjectId(dataset['_id'])) + "_full_img"+".json"):
            f = open ('/critical_data/regions/'+str(ObjectId(dataset['_id'])) + "_full_img"+".json", "r")
            a = json.loads(f.read())
            full_img = a['full_img']
            print("GOTTTTTTTTTTTTTTTTTTT")
            f.close()
        else:
            full_img = None
            
        
    except:
        message = "error in full_img/regions not set"
        status_code = 400
        return message,status_code
    
    if full_img is None:
        message = "error in full_img/regions not set"
        status_code = 400
        return message,status_code

    



    ########### load the model to the memory 

    """
    base_path =  os.path.join('/critical_data/')
    if oem_number is None:
        this_model_pth = str(jig_type) 
    else:
        this_model_pth = str(jig_type) + str(oem_number) 

    dir_path = os.path.join(base_path,this_model_pth)

    weight_pth = os.path.join(dir_path,'weights')

    inference_grp_pth = os.path.join(weight_pth,'saved_model')
    inference_grp_pth = os.path.join(inference_grp_pth,'saved_model.pb')
    labelmap_pth = os.path.join(weight_pth,'labelmap.txt')

    PATH_TO_CFG = None
    PATH_TO_CKPT = None
    PATH_TO_LABELS = None
    detection_model,category_index = load_model_to_memory(PATH_TO_CFG,PATH_TO_CKPT,PATH_TO_LABELS)
    """

    #if jig_type == "GVM":
    #    detection_graph,sess,accuracy_threshold  = load_gvm_model()
    #else:
    #    detection_graph,sess,accuracy_threshold  = load_gvx_model()

    #####################################

    #final_dct = {}


    def regions_crop_pred(regions,rch,r_key,final_dct,port):
        global frame
        global crp
        t1 = time.time()
        frame  = rch.get_json(r_key)
        t2 = time.time()
        print('time taken to get frame from redis  ::  ::  ::  :: ---------    '+ str(t2-t1))
        

        height,width,c = frame.shape

        #height = height*3
        #width = width*3

        def resize_crop(img):
            scale_percent = 40 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height) 
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS64) 
            return resized



        for j in regions:
            #print("checking regions : : : " , j)
            x = float(j["x"])
            y = float(j["y"])
            w = float(j["w"])
            h = float(j["h"])
            #print(x,y,w,h)
            x0 = int(x * width)
            y0 = int(y * height)
            x1 = int(((x+w) * width))
            y1 = int(((y+h) * height))
            #print(x0,y0,x1,y1)
            label = j["cls"]
            cords = [x0,y0,x1,y1]
            import uuid
            unique_id = str(uuid.uuid4())

            #perform crop
            t1 = time.time()
            crp = frame[y0:y1,x0:x1].copy()
            #crp = resize_crop(crp)
            t2 = time.time()
            print('time taken to get crop            ::  ::  ::  :: ---------    '+ str(t2-t1))
            
            
            #cv2.imwrite('/critical_data/tmpcrops/'+unique_id+'.jpg',crp)
            #print('/critical_data/tmpcrops/'+unique_id+'.jpg')

            if jig_type == "GVM":
                print("going into prediction")
                print(crp.shape)
                t1 = time.time()
                ret = get_pred_tf_serving(crp,port,"GVM",gvm_num_classes,gvm_labelmap_pth)
                scores,boxes,objects,accuracy = ret['scores'], ret['boxes'], ret['objects'], ret['accuracy']
                t2 = time.time()
                print('time taken to predict            ::  ::  ::  :: ---------    '+ str(t2-t1))
                #scores,boxes,objects,accuracy = detect_gvm_frame(crp,detection_graphm,sessm,accuracy_thresholdm,category_indexm)
                #print('done with prediction')
            else:
                print("going into prediction")
                print(crp.shape)
                t1 = time.time()
                ret = get_pred_tf_serving(crp,port,"GVX",gvx_num_classes,gvx_labelmap_pth)
                scores,boxes,objects,accuracy = ret['scores'], ret['boxes'], ret['objects'], ret['accuracy']
                t2 = time.time()
                print('time taken to predict            ::  ::  ::  :: ---------    '+ str(t2-t1))
                #scores,boxes,objects,accuracy = detect_gvm_frame(crp,detection_graphm,sessm,accuracy_thresholdm,category_indexm)
                #print('done with prediction')
                #scores,boxes,objects,accuracy = detect_gvx_frame(crp,detection_graphx,sessx,accuracy_thresholdx,category_indexx)

            #send this crop to inference and get back predicted text detection (region label : predicted label)
            #detections, predictions_dict, shapes = crop_infer(crp)

            #print(scores)
            #print(boxes)
            #print(objects)

            if len(objects) == 0:
                #no predictions
                final_dct[label] = None

            elif len(objects) == 1:
                #one prediction (maybe true or maybe false prediction)
                predicted_obj = str(objects[0])
                if '_' in predicted_obj:
                    predicted_obj = str(predicted_obj.split('_')[0])
                final_dct[label] = predicted_obj

            else:
                #multiple predictions (sort acc to accuracy and take highest acc)
                original_lst = accuracy.copy()
                if len(accuracy) > 0:
                    accuracy.sort(reverse = True)

                first_ele = accuracy[0]

                idx_acc = original_lst.index(first_ele)
                predicted_obj = str(objects[idx_acc])
                if '_' in predicted_obj:
                    predicted_obj = str(predicted_obj.split('_')[0])
                final_dct[label] = predicted_obj

                
                
        return final_dct




    #write a while true loop : if final dict match with kanban  or manual pass by admin using inspection_id
    t_start_inspection = time.time()

    loop_idx_for_del = 0

    def do_del(cl_obj):
        try:
            del cl_obj
        except Exception as e:
            print("del ops")
            print(e)
    retry_list = []




    while(True):
        if retry_list != []:

            var = str(ObjectId(dataset['_id'])) + "_paused"
            is_paused = rch.get_json(var)
            #rch.set_json({var:p1_lst})

            if is_paused:
                #print("i am paused")
                continue
                


        print("Testing here")
        try:
            mp = MongoHelper().getCollection('INSPECTION')
        except Exception as e:
            message = "Cannot connect to db"
            status_code = 500
            return message,status_code


        try:
            dataset = mp.find_one({'_id' : ObjectId(inspection_id)})
            if dataset is None:
                message = "Inspection not found in inspection collection"
                status_code = 404
                return message,status_code

        except Exception as e:
            message = "Invalid inspection ID"
            status_code = 400
            return message,status_code


        is_manual_pass = dataset['is_manual_pass']
        is_reject = dataset['is_reject']

        if is_manual_pass is True:
            print("manual pass is true")
            break
        if is_reject is True:
            print("part rejection is true")
            break
        

        
        final_dct = {}
        #print(full_img)
        #print(cam_list)
        #print(key_list)

        def get_pred_extreme_left_camera(p1_lst,jig_type):
            for cam,r_key in zip(cam_list,key_list):
                if cam == "extreme_left_camera":
                    for f in full_img:
                        if f['cam_name'] == 'extreme_left_camera':
                            try:
                                regions = f['regions']
                                if regions != "":
                                    #print('line 399')
                                    print("inside extreme left cam")
                                    if jig_type == "GVM":
                                        p1_lst = regions_crop_pred(regions,rch,r_key,p1_lst,8501)
                                    elif jig_type == "GVX":
                                        p1_lst = regions_crop_pred(regions,rch,r_key,p1_lst,8506)
                                    print("&&&&&&&&&&&&&&&")   
                                    print(p1_lst)
                                    print("&&&&&&&&&&&&&&&")
                                    var = str(inspection_id) + "_cam1"
                                    rch.set_json({var:p1_lst})
                            except Exception as e:
                                print("region not defined in extreme left camera:" + str(e) )
                                pass
                  

        def get_pred_left_camera(p2_lst,jig_type):
            for cam,r_key in zip(cam_list,key_list):
                if cam == "left_camera":
                    for f in full_img:
                        if f['cam_name'] == 'left_camera':
                            try:
                                regions = f['regions']
                                if regions != "":
                                    print("inside left cam")
                                    if jig_type == "GVM":
                                        p2_lst = regions_crop_pred(regions,rch,r_key,p2_lst,8502)
                                    elif jig_type == "GVX":
                                        p2_lst = regions_crop_pred(regions,rch,r_key,p2_lst,8507)
                                    var = str(inspection_id) + "_cam2"
                                    rch.set_json({var:p2_lst})
                            except Exception as e:
                                print("region not defined in left camera:" + str(e) )
                                pass

        def get_pred_middle_camera(p3_lst,jig_type):
            for cam,r_key in zip(cam_list,key_list):
                if cam == "middle_camera":
                    for f in full_img:
                        if f['cam_name'] == 'middle_camera':
                            try:
                                regions = f['regions']
                                if regions != "":
                                    print("inside  middle cam")
                                    if jig_type == "GVM":
                                        p3_lst = regions_crop_pred(regions,rch,r_key,p3_lst,8503)
                                    elif jig_type == "GVX":
                                        p3_lst = regions_crop_pred(regions,rch,r_key,p3_lst,8508)
                                    var = str(inspection_id) + "_cam3"
                                    rch.set_json({var:p3_lst})
                            except Exception as e:
                                print("region not defined in middle camera:" + str(e) )
                                pass  

        def get_pred_right_camera(p4_lst,jig_type):
            for cam,r_key in zip(cam_list,key_list):
                if cam == "right_camera":
                    for f in full_img:
                        if f['cam_name'] == 'right_camera':
                            try:
                                regions = f['regions']
                                if regions != "":
                                    print("inside  right cam")
                                    if jig_type == "GVM":
                                        p4_lst = regions_crop_pred(regions,rch,r_key,p4_lst,8504)
                                    elif jig_type == "GVX":
                                        p4_lst = regions_crop_pred(regions,rch,r_key,p4_lst,8509)
                                    var = str(inspection_id) + "_cam4"
                                    rch.set_json({var:p4_lst})
                            except Exception as e:
                                print("region not defined in right camera:" + str(e) )
                                pass  

        def get_pred_extreme_right_camera(p5_lst,jig_type):
            for cam,r_key in zip(cam_list,key_list):
                if cam == "extreme_right_camera":
                    for f in full_img:
                        if f['cam_name'] == 'extreme_right_camera':
                            try:
                                regions = f['regions']
                                if regions != "":
                                    print("inside extreme right cam")
                                    if jig_type == "GVM":
                                        p5_lst = regions_crop_pred(regions,rch,r_key,p5_lst,8505)
                                    elif jig_type == "GVX":
                                        p5_lst = regions_crop_pred(regions,rch,r_key,p5_lst,8510)
                                    var = str(inspection_id) + "_cam5"
                                    rch.set_json({var:p5_lst})
                            except Exception as e:
                                print("region not defined in extreme right camera:" + str(e) )
                                pass
        t1 = time.time()
        #check if something is running - if yes check what it is ? if selected and running match then pass else kill running and start selected

        def launch_gvm_containers():
            t1 = time.time()

            command1 = "docker container run --gpus all -p 8501:8501 -v " + "\'" + gvm_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            command2 = "docker container run --gpus all -p 8502:8501 -v " + "\'" + gvm_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            command3 = "docker container run --gpus all -p 8503:8501 -v " + "\'" + gvm_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            command4 = "docker container run --gpus all -p 8504:8501 -v " + "\'" + gvm_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            command5 = "docker container run --gpus all -p 8505:8501 -v " + "\'" + gvm_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            passwd = '123456789'
            
            subprocess.Popen(command1,shell=True)
            subprocess.Popen(command2,shell=True)
            subprocess.Popen(command3,shell=True)
            subprocess.Popen(command4,shell=True)
            subprocess.Popen(command5,shell=True)
            
            

            time.sleep(10)
            
            print("gvm containers launched")
            t2 = time.time()
            print('gvm docker making it up and running time is            ::  ::  ::  :: ---------    '+ str(t2-t1))

        
        def launch_gvx_containers():
            t1 = time.time()
            command1 = "docker container run --gpus all -p 8506:8501 -v " + "\'" + gvx_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            command2 = "docker container run --gpus all -p 8507:8501 -v " + "\'" + gvx_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            command3 = "docker container run --gpus all -p 8508:8501 -v " + "\'" + gvx_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            command4 = "docker container run --gpus all -p 8509:8501 -v " + "\'" + gvx_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            command5 = "docker container run --gpus all -p 8510:8501 -v " + "\'" + gvx_saved_model_pth + "\'" + ":/models/saved_model -e MODEL_NAME=saved_model tensorflow/serving:1.14.0-gpu  --per_process_gpu_memory_fraction=0.15"
            passwd = '123456789'
            subprocess.Popen(command1,shell=True)
            subprocess.Popen(command2,shell=True)
            subprocess.Popen(command3,shell=True)
            subprocess.Popen(command4,shell=True)
            subprocess.Popen(command5,shell=True)
            
            
            time.sleep(10)
            print("gvx containers launched")
            t2 = time.time()
            print('gvx docker making it up and running time is            ::  ::  ::  :: ---------    '+ str(t2-t1))


        proc = subprocess.run(['docker','container','ls'],check=True,stdout=PIPE)
        container_ids = proc.stdout.split()
        gvx_gvm_lst = []
        for i in container_ids:
            i = i.decode("utf-8")
            if '0.0.0.0:8501' in i:
                gvx_gvm_lst.append("8501")
            elif '0.0.0.0:8502' in i :
                gvx_gvm_lst.append("8502")
            elif '0.0.0.0:8503' in i:
                gvx_gvm_lst.append("8503")
            elif '0.0.0.0:8504' in i:
                gvx_gvm_lst.append("8504")
            elif '0.0.0.0:8505' in i:
                gvx_gvm_lst.append("8505")
            elif '0.0.0.0:8506' in i :
                gvx_gvm_lst.append("8506")
            elif '0.0.0.0:8507' in i:
                gvx_gvm_lst.append("8507")
            elif '0.0.0.0:8508' in i:
                gvx_gvm_lst.append("8508")
            elif '0.0.0.0:8509' in i:
                gvx_gvm_lst.append("8509")
            elif '0.0.0.0:8510' in i:
                gvx_gvm_lst.append("8510")



        if len(gvx_gvm_lst) == 0:
            #no containers running # launch appropriate container
            print("no containers are running")

            if jig_type == "GVM":
                #launch gvm dockers 
                launch_gvm_containers()
                

            elif jig_type == "GVX":
                #launch gvx dockers
                launch_gvx_containers()
                

        else:
            #some container are running check which one
            if '8501' in gvx_gvm_lst and '8502' in gvx_gvm_lst and '8503' in gvx_gvm_lst and '8504' in gvx_gvm_lst and '8505' in gvx_gvm_lst:
                #gvm is running 
                print("gvm is running")

                if jig_type == "GVM":
                    pass
                elif jig_type == "GVX":
                    print("killing gvm docker to launch gvx")
                    t1 = time.time()
                    #subprocess.Popen("sudo systemctl restart docker",shell=True).wait()
                    proc = subprocess.run(['docker','ps','-aq'],check=True,stdout=PIPE,encoding='ascii')
                    container_ids = proc.stdout.strip().split()
                    if container_ids:
                        subprocess.run(['docker','stop']+container_ids,check=True)
                        subprocess.run(['docker','rm']+container_ids,check=True)
                    t2 = time.time()
                    print('docker restarting time took             ::  ::  ::  :: ---------    '+ str(t2-t1))
                    print("killed....")
                    launch_gvx_containers()


            elif '8506' in gvx_gvm_lst and '8507' in gvx_gvm_lst and '8508' in gvx_gvm_lst and '8509' in gvx_gvm_lst and '8510' in gvx_gvm_lst:
                #gvx is running 
                print("gvx is running")

                if jig_type == "GVX":
                    pass
                elif jig_type == "GVM":
                    print("killing gvx docker to launch gvm")
                    t1 = time.time()
                    #subprocess.Popen("sudo systemctl restart docker",shell=True).wait()
                    proc = subprocess.run(['docker','ps','-aq'],check=True,stdout=PIPE,encoding='ascii')
                    container_ids = proc.stdout.strip().split()
                    if container_ids:
                        subprocess.run(['docker','stop']+container_ids,check=True)
                        subprocess.run(['docker','rm']+container_ids,check=True)
                    t2 = time.time()
                    print('docker restarting time took             ::  ::  ::  :: ---------    '+ str(t2-t1))
                    print("killed....")
                    launch_gvm_containers()


        p1_lst = {}
        p2_lst = {}
        p3_lst = {}
        p4_lst = {}
        p5_lst = {}
        
        var = str(inspection_id) + "_cam1"
        rch.set_json({var:p1_lst})
        var = str(inspection_id) + "_cam2"
        rch.set_json({var:p2_lst})
        var = str(inspection_id) + "_cam3"
        rch.set_json({var:p3_lst})
        var = str(inspection_id) + "_cam4"
        rch.set_json({var:p4_lst})
        var = str(inspection_id) + "_cam5"
        rch.set_json({var:p5_lst})
        
        P1 = Process(target=get_pred_extreme_left_camera,args=(p1_lst,jig_type,))
        P2 = Process(target=get_pred_left_camera,args=(p2_lst,jig_type,))
        P3 = Process(target=get_pred_middle_camera,args=(p3_lst,jig_type,))
        P4 = Process(target=get_pred_right_camera,args=(p4_lst,jig_type,))
        P5 = Process(target=get_pred_extreme_right_camera,args=(p5_lst,jig_type,))
        
        P1.start()
        P2.start()
        P3.start()
        P4.start()
        P5.start()

        P1.join()
        P2.join()
        P3.join()
        P4.join()
        P5.join()
        
        t2 = time.time()
        print('TIMEEEEEEEEEEEEEEEEEEEEEEEEEEE            ::  ::  ::  :: ---------    '+ str(t2-t1))
        
        var = str(inspection_id) + "_cam1"
        p1_lst = rch.get_json(var)
        var = str(inspection_id) + "_cam2"
        p2_lst = rch.get_json(var)
        var = str(inspection_id) + "_cam3"
        p3_lst = rch.get_json(var)
        var = str(inspection_id) + "_cam4"
        p4_lst = rch.get_json(var)
        var = str(inspection_id) + "_cam5"
        p5_lst = rch.get_json(var)
        
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(p1_lst)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        final_dct = {}
        final_dct.update(p1_lst)
        final_dct.update(p2_lst)
        final_dct.update(p3_lst)
        final_dct.update(p4_lst)
        final_dct.update(p5_lst)
	
        

        t1 = time.time()
        # compare the final_dct with the actual kanban if all match break else continue  (keep updating the inspection_id key of redis with matched values)
        
        region_pass_fail = []


        def populate_results(pos,k,value):
            
            if k['part_type'] == 'IGBT':

                # if no predictions on region (either he kept part which isnt trained or network hasn't learnt that part well or he hasn't kept anythin at all) if no detections -yellow
                if value is None:
                    region_pass_fail.append( {"position":k['position'],"part_number":k['part_number'],"status":False,"result_part_number":None,"color":"yellow"} )
                else:
                    #if there is a prediction (it can be right or wrong prediction) - 
                    #HAS_PART = True
                    for part in k['part_number']:
                        if str(value) in part: #if model gave right prediction and right part is placed in location - green
                            HAS = False
                            indexx = 0
                            for indivi in region_pass_fail:
                                #print("************************************************************")
                                #print(str(indivi["position"]))
                                #print("\n")
                                #print(str(k['position']))
                                #print('\n')
                                #print(str(indivi["position"]) == str(k['position']))
                                #print("************************************************************")
                                if str(indivi["position"]) == str(k['position']):
                                    HAS = True
                                    break
                                indexx = indexx+1
                            if HAS is True:
                                region_pass_fail[indexx] = {"position":k['position'],"part_number":k['part_number'],"status":True,"result_part_number":str(value),"color":"green"}
                                break
                            else:
                                region_pass_fail.append( {"position":k['position'],"part_number":k['part_number'],"status":True,"result_part_number":str(value),"color":"green"} )
                                break
                        else: #if it cant find in array (operator placed wrong trained part in wrong position or our model gave false prediction) - red
                            #check if already exist - if pos exist in region_pass_fail then dont append else append
                            HAS = False
                            for indivi in region_pass_fail:
                                #print("************************************************************")
                                #print(str(indivi["position"]))
                                #print("\n")
                                #print(str(k['position']))
                                #print('\n')
                                #print(str(indivi["position"]) == str(k['position']))
                                #print("************************************************************")
                                if str(indivi["position"]) == str(k['position']):
                                    HAS = True
                                    break
                            if HAS is True:
                                pass
                            else:
                                region_pass_fail.append( {"position":k['position'],"part_number":k['part_number'],"status":False,"result_part_number":str(value),"color":"red"} )

            elif k['part_type'] == 'THERMOSTAT':
                #thermostat logic

                # if no predictions on region (either he kept part which isnt trained or network hasn't learnt that part well or he hasn't kept anythin at all) if no detections
                if value is None: #yellow
                    region_pass_fail.append( {"position":k['position'],"part_number":k['part_number'],"status":False,"result_part_number":None,"color":"yellow"} )
                else:
                    #if there is a prediction (it can be right or wrong prediction) - 
                    if value == "thermostat":#if model gave right prediction and right part is placed in location - green
                        region_pass_fail.append( {"position":k['position'],"part_number":k['part_number'],"status":True,"result_part_number":str(value),"color":"green"} )

                    else: #if it cant find in array (operator placed wrong trained part in wrong position or our model gave false prediction) - red 
                        region_pass_fail.append( {"position":k['position'],"part_number":k['part_number'],"status":False,"result_part_number":str(value),"color":"red"} )
            else:
                region_pass_fail.append( {"position":k['position'],"part_number":k['part_number'],"status":False,"result_part_number":None,"color":"yellow"} )
        pos_counter = {}
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(final_dct)
        print("###############################")
        for k in kanban:
         # for each prediction  final_dct = {"region1":"None","region2":"k75t60"}
            #print(key, value)
             # for each actual value
            #    print("kanban_key", k)
            for key,value in final_dct.items():
                #print(k['position'],key, value)
                pos_key = int(key.replace('region',''))
                if k['position'] == pos_key:
                    populate_results(key,k,value)
                    pos_counter[k['position']] = key
                #if k['position'] == 1 and key == "region1":
                #    populate_results(key,k,value)
                #elif k['position'] == 2 and key == "region2":
                #    populate_results(key,k,value)
                #elif k['position'] == 3 and key == "region3":
                #    populate_results(key,k,value)
                #elif k['position'] == 4 and key == "region4":
                #    populate_results(key,k,value)
                #elif k['position'] == 5 and key == "region5":
                #    populate_results(key,k,value)
                #elif k['position'] == 6 and key == "region6":
                #    populate_results(key,k,value)
                #elif k['position'] == 7 and key == "region7":
                #    populate_results(key,k,value)
                #elif k['position'] == 8 and key == "region8":
                #    populate_results(key,k,value)
                #elif k['position'] == 9 and key == "region9":
                #    populate_results(key,k,value)
                #elif k['position'] == 10 and key == "region10":
                #    populate_results(key,k,value)
                #elif k['position'] == 11 and key == "region11":
                #    populate_results(key,k,value)
                #elif k['position'] == 12 and key == "region12":
                #    populate_results(key,k,value)
                #elif k['position'] == 13 and key == "region13":
                #    populate_results(key,k,value)
                #elif k['position'] == 14 and key == "region14":
                #    populate_results(key,k,value)
                #elif k['position'] == 15 and key == "region15":
                #    populate_results(key,k,value)
                #elif k['position'] == 16 and key == "region16":
                #    populate_results(key,k,value)
                #elif k['position'] == 17 and key == "region17":
                #    populate_results(key,k,value)
                #elif k['position'] == 18 and key == "region18":
                #    populate_results(key,k,value)
                #elif k['position'] == 19 and key == "region19":
                #    populate_results(key,k,value)
                #elif k['position'] == 20 and key == "region20":
                #    populate_results(key,k,value)
                else:
                    pass
        for k in kanban:
            if k['position'] not in pos_counter:
                populate_results('region'+str(k['position']),k,None)

        #print(region_pass_fail)
        region_pass_fail = sorted(region_pass_fail, key = lambda i: i['position'])
        
        
        #print(pos_counter)
        #vendor match  - correct the vendors --- [[1,2,5],[3,4],[7,8,9],[10,11,12]] convert to [[p100,p100,p100],[abcd,abcd],[123,123,123],[l,l,l]]
        """
        tmp_vendor = []
        for vendd in vendor_match:
            k = vendd.replace(",","")
            k = int(k)
            tmp_vendor.append([k])

        last = []
        first = []
        for i in tmp_vendor:
            o = 0
            while(o<len(str(i))):
                last.append(int(str(i)[o]))
                o=o+1
            first.append(last)
            last = []

        vendor_match = []
        vendor_match = first.copy()
        """
        e = []
        f=[]
        print(vendor_match)
        for v in vendor_match:
            
            for h in v.split(','):
                e.append(int(h))
            f.append(e)
            e  = []
                

        vendor_match = f.copy()
        value_region_acc_to_vendors = []
        sub_list = []
        for pos_list in vendor_match:                     
            for pos in pos_list:
                for region_pf in region_pass_fail:
                    if region_pf['status'] == True: #if it passed above checks

                        if region_pf['position'] == pos :
                            sub_list.append(str(region_pf['result_part_number']))

            value_region_acc_to_vendors.append(sub_list)
            sub_list = []

        
        IS_EVEN = True

        if len(value_region_acc_to_vendors) == len(vendor_match):
            i = 0
            while( i<len(value_region_acc_to_vendors)-1 ):

                if len(vendor_match[i]) == len(value_region_acc_to_vendors[i]):
                    pass 
                else:
                    IS_EVEN = False

                i=i+1
         
        print(value_region_acc_to_vendors)
        print("\n")
        print(vendor_match)
        print("ISEVEN IS ")
        print(IS_EVEN)

        if IS_EVEN:

            def chkList(lst): 
                res = False
                if len(lst) < 0 : 
                    res = True
                res = all(ele == lst[0] for ele in lst) 
                
                if(res): 
                    return True
                else: 
                    return False


            print("entering check...")
            for ven,vendor_m in zip(value_region_acc_to_vendors,vendor_match):
            
                is_match = chkList(ven)

                if is_match:
                    #all are green
                    print("all vendors are same - pass")
                    pass


                if is_match is False:
                    print("all vendors are not same - fail")
                    
                    #check if all are unique (all igbt are different - make all red)
                    print(ven)
                    print(len(set(ven)))
                    print(len(ven))
                    if(len(set(ven)) == len(ven)):
                        print("all vendors are unique all fail")
                        #all are unique 
                        for positions in vendor_m:

                            for r_p_f in region_pass_fail:

                                if r_p_f['position'] == positions:

                                    #make red in index of that
                                    idx = region_pass_fail.index(r_p_f)
                                    region_pass_fail[idx] = {"position":r_p_f['position'],"part_number":r_p_f['part_number'],"status":False,"result_part_number":r_p_f['result_part_number'],"color":"red","message":"vendor match failed"}
                    else:
                        print("some are same and some are diff")
                        #some are same some are different
                        #append to dict
                        freq = {}

                        for item in ven: 
                            if (item in freq): 
                                freq[item] += 1
                            else: 
                                freq[item] = 1
                        print("frquency wihout sorting")
                        print(freq)

                        #check if sets are same - if yes retain one set and make other sets as red 
                        set_value = []

                        for key,value in freq.items():
                            set_value.append(value)

                        is_matched = chkList(set_value)
                        print("setvalue is")
                        print(set_value)
                        if is_matched:
                            print("sets are of same freq")
                            # sets are of same frequency --  retain one set and make other sets as red
                            
                            for key,value in freq.items():
                                first_key = str(key)
                                break

                            for positions,pred_val in zip(vendor_m,ven):

                                for r_p_f in region_pass_fail:

                                    if r_p_f['position'] == positions and str(pred_val) != first_key:

                                        #make red in index of that
                                        idx = region_pass_fail.index(r_p_f)
                                        region_pass_fail[idx] = {"position":r_p_f['position'],"part_number":r_p_f['part_number'],"status":False,"result_part_number":r_p_f['result_part_number'],"color":"red","message":"vendor match failed"}

                        else:
                            print("sets are of different freq")
                            #sets are not same - find greatest set using sort - keep that as green and make other sets as red
                            sorted_freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1])}
                            print("sorted freq issssss")
                            print(sorted_freq)
                            for key,value in sorted_freq.items():
                                first_key = str(key)
                                
                            print("first key is ")
                            print(first_key)
                            
                            for positions,pred_val in zip(vendor_m,ven):

                                for r_p_f in region_pass_fail :
                                    
                                    if r_p_f['position'] == positions and str(pred_val) != first_key:
                                    
                                        idx = region_pass_fail.index(r_p_f)
                                        region_pass_fail[idx] = {"position":r_p_f['position'],"part_number":r_p_f['part_number'],"status":False,"result_part_number":r_p_f['result_part_number'],"color":"red","message":"vendor match failed"}

        
            t2 = time.time()
            #print('time taken to execute comparision logic....            ::  ::  ::  :: ---------    '+ str(t2-t1))


      

            
        #write to redis : key is inspection_id and value is region_pass_fail
        rch.set_json({inspection_id:region_pass_fail})
        var = str(inspection_id) + "_result"
        rch.set_json({var:"fail"})
        
        cycle_time_key = str(inspection_id) + "_cycletime"
        rch.set_json({cycle_time_key: time.time() - t_start_inspection })
        #with this pass or fail


        #compare all if all status is true then break
        IS_PROCESS_END = True
        for final_check in region_pass_fail:
            if final_check['status'] is True:
                pass
            else:
                IS_PROCESS_END = False

        if IS_PROCESS_END:
            break

        retry_list.append(region_pass_fail)
        print("LENGTH!!")
        print(len(retry_list))
        var = str(inspection_id) + "_retry_array"
        rch.set_json({var:retry_list})

        if len(retry_list) == 0:
            pass
        elif len(retry_list) >= 1:

            try:
                rch = CacheHelper()
                var = str(ObjectId(dataset['_id'])) + "_paused"
                #is_paused = rch.get_json(var)
                rch.set_json({var:True})
            except:
                return "error setting redis",400



        
        try: do_del(mp)
        except: pass
        try: do_del(message)
        except: pass
        try: do_del(status_code)
        except: pass
        try: do_del(dataset)
        except: pass
        try: do_del(is_manual_pass)
        except: pass
        try: do_del(cam_list)
        except: pass
        try: do_del(key_list)
        except: pass
        try: do_del(region_pass_fail)
        except: pass
        try: do_del(e)
        except: pass
        try: do_del(f)
        except: pass
        try: do_del(final_dct)
        except: pass
        try: do_del(p1_lst)
        except: pass   
        try: do_del(p2_lst)
        except: pass
        try: do_del(p3_lst)
        except: pass
        try: do_del(p4_lst)
        except: pass
        try: do_del(p5_lst)
        except: pass
        try: do_del(pos_counter)
        except: pass
        try: do_del(t1)
        except: pass
        try: do_del(t2)
        except: pass
        try: do_del(P1)
        except: pass
        try: do_del(P2)
        except: pass
        try: do_del(P3)
        except: pass
        try: do_del(P4)
        except: pass
        try: do_del(P5)
        except: pass   
        try: do_del(var)
        except: pass
        try: do_del(key)
        except: pass
        try: do_del(value)
        except: pass
        try: do_del(pos_key)
        except: pass
        try: do_del(pos_list)
        except: pass
        try: do_del(k)
        except: pass
        try: do_del(pos)
        except: pass
        try: do_del(vendor_match)
        except: pass
        try: do_del(vendor_m)
        except: pass
        try: do_del(value_region_acc_to_vendors)
        except: pass
        try: do_del(sub_list)
        except: pass
        try: do_del(r_p_f)
        except: pass   

        try: gc.collect()
        except: pass


        
    #outside loop : update inspection id collection with pass and end time.


    var = str(inspection_id) + "_result"
    rch.set_json({var:"pass"})

    try:
        mp = MongoHelper().getCollection('INSPECTION')
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code


    try:
        dataset = mp.find_one({'_id' : ObjectId(inspection_id)})
        if dataset is None:
            message = "Inspection not found in inspection collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid inspection ID"
        status_code = 400
        return message,status_code

    dataset['completedAt'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    completedAt = datetime.datetime.strptime(dataset['completedAt'], '%Y-%m-%d %H:%M:%S')
    createdAt = datetime.datetime.strptime(dataset['createdAt'], '%Y-%m-%d %H:%M:%S')
    duration = completedAt - createdAt
    dataset['duration'] = str(duration)
    dataset['status'] = 'completed'
    dataset['is_compleated'] = True
    
    mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  dataset})

    #do gc collect here
    gc.collect()




#get method
def get_current_inspection_details_utils(inspection_id):
    #{eval_data}

    #inspection_id =  data['inspection_id']
    if inspection_id is None:
        message = "inspection_id not provided"
        status_code = 400
        return message,status_code

    try:
        mp = MongoHelper().getCollection('INSPECTION')
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    try:
        dataset = mp.find_one({'_id' : ObjectId(inspection_id)})
        if dataset is None:
            message = "Inspection not found in inspection collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid inspection ID"
        status_code = 400
        return message,status_code
    samp = {}
    if True:
        #samp = {}
        rch = CacheHelper()
        details = rch.get_json(str(inspection_id))
        var = str(inspection_id) + "_result"
        result = rch.get_json(var)
        
        new_details = {}
        blank = []
        if details is not None:
            print("details is not None")
            print(details)
            for evalu in details:
                #print(str(dataset['jig_details']['oem_number'])) 
                if str(dataset['jig_details']['oem_number']) == "0P3621-HS7000":
                    if evalu['position'] == 1:
                        evalu['static'] = "Q7018"
                        blank.append(evalu)
                    elif evalu['position'] == 2:
                        evalu['static'] = "Q7019"
                        blank.append(evalu)
                    elif evalu['position'] == 3:
                        evalu['static'] = "Q7022"
                        blank.append(evalu)
                    elif evalu['position'] == 4:
                        evalu['static'] = "Q7023"
                        blank.append(evalu)
                    elif evalu['position'] == 5:
                        evalu['static'] = "Q7012"
                        blank.append(evalu)
                    elif evalu['position'] == 6:
                        evalu['static'] = "Q7013"
                        blank.append(evalu)
                    elif evalu['position'] == 7:
                        evalu['static'] = "Q7014"
                        blank.append(evalu)
                    elif evalu['position'] == 8:
                        evalu['static'] = "Q7015"
                        blank.append(evalu)
                    elif evalu['position'] == 9:
                        evalu['static'] = "SW7001"
                        blank.append(evalu)
                    elif evalu['position'] == 10:
                        evalu['static'] = "Q7004"
                        blank.append(evalu)
                    elif evalu['position'] == 11:
                        evalu['static'] = "Q7005"
                        blank.append(evalu)
                    elif evalu['position'] == 12:
                        evalu['static'] = "Q7006"
                        blank.append(evalu)
                    elif evalu['position'] == 13:
                        evalu['static'] = "Q7007"
                        blank.append(evalu)
                    elif evalu['position'] == 14:
                        evalu['static'] = "Q7026"
                        blank.append(evalu)
                    elif evalu['position'] == 15:
                        evalu['static'] = "Q7027"
                        blank.append(evalu)
                    elif evalu['position'] == 16:
                        evalu['static'] = "Q7030"
                        blank.append(evalu)
                    elif evalu['position'] == 17:
                        evalu['static'] = "Q7031"
                        blank.append(evalu)
                    elif evalu['position'] == 18:
                        evalu['static'] = "Q7032"
                        blank.append(evalu)
                    elif evalu['position'] == 19:
                        evalu['static'] = "Q7033"
                        blank.append(evalu)
                        
                elif str(dataset['jig_details']['oem_number']) == "0P3621-HS7001":
                    if evalu['position'] == 1:
                        evalu['static'] = "Q7034"
                        blank.append(evalu)
                    elif evalu['position'] == 2:
                        evalu['static'] = "Q7035"
                        blank.append(evalu)
                    elif evalu['position'] == 3:
                        evalu['static'] = "Q7028"
                        blank.append(evalu)
                    elif evalu['position'] == 4:
                        evalu['static'] = "Q7029"
                        blank.append(evalu)
                    elif evalu['position'] == 5:
                        evalu['static'] = "Q7024"
                        blank.append(evalu)
                    elif evalu['position'] == 6:
                        evalu['static'] = "Q7025"
                        blank.append(evalu)
                    elif evalu['position'] == 7:
                        evalu['static'] = "Q7000"
                        blank.append(evalu)
                    elif evalu['position'] == 8:
                        evalu['static'] = "Q7001"
                        blank.append(evalu)
                    elif evalu['position'] == 9:
                        evalu['static'] = "Q7002"
                        blank.append(evalu)
                    elif evalu['position'] == 10:
                        evalu['static'] = "Q7003"
                        blank.append(evalu)
                    elif evalu['position'] == 11:
                        evalu['static'] = "SW7000"
                        blank.append(evalu)
                    elif evalu['position'] == 12:
                        evalu['static'] = "Q7008"
                        blank.append(evalu)
                    elif evalu['position'] == 13:
                        evalu['static'] = "Q7009"
                        blank.append(evalu)
                    elif evalu['position'] == 14:
                        evalu['static'] = "Q7010"
                        blank.append(evalu)
                    elif evalu['position'] == 15:
                        evalu['static'] = "Q7011"
                        blank.append(evalu)
                    elif evalu['position'] == 16:
                        evalu['static'] = "Q7020"
                        blank.append(evalu)
                    elif evalu['position'] == 17:
                        evalu['static'] = "Q7021"
                        blank.append(evalu)
                    elif evalu['position'] == 18:
                        evalu['static'] = "Q7016"
                        blank.append(evalu)
                    elif evalu['position'] == 19:
                        evalu['static'] = "Q7017"
                        blank.append(evalu)

                elif str(dataset['jig_details']['oem_number']) == "0P3623-HS7100":
                    if evalu['position'] == 1:
                        evalu['static'] = "Q7122"
                        blank.append(evalu)
                    elif evalu['position'] == 2:
                        evalu['static'] = "Q7123"
                        blank.append(evalu)
                    elif evalu['position'] == 3:
                        evalu['static'] = "Q7118"
                        blank.append(evalu)
                    elif evalu['position'] == 4:
                        evalu['static'] = "Q7119"
                        blank.append(evalu)
                    elif evalu['position'] == 5:
                        evalu['static'] = "Q7120"
                        blank.append(evalu)
                    elif evalu['position'] == 6:
                        evalu['static'] = "Q7121"
                        blank.append(evalu)
                    elif evalu['position'] == 7:
                        evalu['static'] = "Q7108"
                        blank.append(evalu)
                    elif evalu['position'] == 8:
                        evalu['static'] = "D7127"
                        blank.append(evalu)
                    elif evalu['position'] == 9:
                        evalu['static'] = "Q7109"
                        blank.append(evalu)
                    elif evalu['position'] == 10:
                        evalu['static'] = "SW7101"
                        blank.append(evalu)
                    elif evalu['position'] == 11:
                        evalu['static'] = "Q7112"
                        blank.append(evalu)
                    elif evalu['position'] == 12:
                        evalu['static'] = "D7129"
                        blank.append(evalu)
                    elif evalu['position'] == 13:
                        evalu['static'] = "Q7113"
                        blank.append(evalu)
                    elif evalu['position'] == 14:
                        evalu['static'] = "D7125"
                        blank.append(evalu)
                    elif evalu['position'] == 15:
                        evalu['static'] = "Q7100"
                        blank.append(evalu)
                    elif evalu['position'] == 16:
                        evalu['static'] = "Q7101"
                        blank.append(evalu)
                    elif evalu['position'] == 17:
                        evalu['static'] = "Q7102"
                        blank.append(evalu)
                    elif evalu['position'] == 18:
                        evalu['static'] = "D7122"
                        blank.append(evalu)
                    elif evalu['position'] == 19:
                        evalu['static'] = "D7123"
                        blank.append(evalu)

                elif str(dataset['jig_details']['oem_number']) == "0P3623-HS7101":
                    if evalu['position'] == 1:
                        evalu['static'] = "D7120"
                        blank.append(evalu)
                    elif evalu['position'] == 2:
                        evalu['static'] = "D7121"
                        blank.append(evalu)
                    elif evalu['position'] == 3:
                        evalu['static'] = "Q7105"
                        blank.append(evalu)
                    elif evalu['position'] == 4:
                        evalu['static'] = "Q7104"
                        blank.append(evalu)
                    elif evalu['position'] == 5:
                        evalu['static'] = "Q7103"
                        blank.append(evalu)
                    elif evalu['position'] == 6:
                        evalu['static'] = "D7124"
                        blank.append(evalu)
                    elif evalu['position'] == 7:
                        evalu['static'] = "Q7111"
                        blank.append(evalu)
                    elif evalu['position'] == 8:
                        evalu['static'] = "D7128"
                        blank.append(evalu)
                    elif evalu['position'] == 9:
                        evalu['static'] = "Q7110"
                        blank.append(evalu)
                    elif evalu['position'] == 10:
                        evalu['static'] = "SW7100"
                        blank.append(evalu)
                    elif evalu['position'] == 11:
                        evalu['static'] = "Q7107"
                        blank.append(evalu)
                    elif evalu['position'] == 12:
                        evalu['static'] = "D7126"
                        blank.append(evalu)
                    elif evalu['position'] == 13:
                        evalu['static'] = "Q7106"
                        blank.append(evalu)
                    elif evalu['position'] == 14:
                        evalu['static'] = "Q7115"
                        blank.append(evalu)
                    elif evalu['position'] == 15:
                        evalu['static'] = "Q7114"
                        blank.append(evalu)
                    elif evalu['position'] == 16:
                        evalu['static'] = "Q7117"
                        blank.append(evalu)
                    elif evalu['position'] == 17:
                        evalu['static'] = "Q7118"
                        blank.append(evalu)
                    elif evalu['position'] == 18:
                        evalu['static'] = "Q7125"
                        blank.append(evalu)
                    elif evalu['position'] == 19:
                        evalu['static'] = "Q7124"
                        blank.append(evalu)

                elif str(dataset['jig_details']['oem_number']) == "0P3647-HS7001":
                    if evalu['position'] == 1:
                        evalu['static'] = "Q7034"
                        blank.append(evalu)
                    elif evalu['position'] == 2:
                        evalu['static'] = "Q7035"
                        blank.append(evalu)
                    elif evalu['position'] == 3:
                        evalu['static'] = "Q7025"
                        blank.append(evalu)
                    elif evalu['position'] == 4:
                        evalu['static'] = "Q7026"
                        blank.append(evalu)
                    elif evalu['position'] == 5:
                        evalu['static'] = "Q7027"
                        blank.append(evalu)
                    elif evalu['position'] == 6:
                        evalu['static'] = "Q7000"
                        blank.append(evalu)
                    elif evalu['position'] == 7:
                        evalu['static'] = "Q7001"
                        blank.append(evalu)
                    elif evalu['position'] == 8:
                        evalu['static'] = "Q7002"
                        blank.append(evalu)
                    elif evalu['position'] == 9:
                        evalu['static'] = "SW7000"
                        blank.append(evalu)
                    elif evalu['position'] == 10:
                        evalu['static'] = "Q7010"
                        blank.append(evalu)
                    elif evalu['position'] == 11:
                        evalu['static'] = "Q7011"
                        blank.append(evalu)
                    elif evalu['position'] == 12:
                        evalu['static'] = "Q7012"
                        blank.append(evalu)
                    elif evalu['position'] == 13:
                        evalu['static'] = "Q7017"
                        blank.append(evalu)
                    elif evalu['position'] == 14:
                        evalu['static'] = "Q7019"
                        blank.append(evalu)
                    elif evalu['position'] == 15:
                        evalu['static'] = "Q7019"
                        blank.append(evalu)

                elif str(dataset['jig_details']['oem_number']) == "0P3648-HS7100":
                    if evalu['position'] == 1:
                        evalu['static'] = "Q7103"
                        blank.append(evalu)
                    elif evalu['position'] == 2:
                        evalu['static'] = "D7124"
                        blank.append(evalu)
                    elif evalu['position'] == 3:
                        evalu['static'] = "Q7104"
                        blank.append(evalu)
                    elif evalu['position'] == 4:
                        evalu['static'] = "D7120"
                        blank.append(evalu)
                    elif evalu['position'] == 5:
                        evalu['static'] = "SW7100"
                        blank.append(evalu)
                    elif evalu['position'] == 6:
                        evalu['static'] = "Q7107"
                        blank.append(evalu)
                    elif evalu['position'] == 7:
                        evalu['static'] = "D7126"
                        blank.append(evalu)
                    elif evalu['position'] == 8:
                        evalu['static'] = "Q7108"
                        blank.append(evalu)
                    elif evalu['position'] == 9:
                        evalu['static'] = "D7127"
                        blank.append(evalu)
                    elif evalu['position'] == 10:
                        evalu['static'] = "Q7109"
                        blank.append(evalu)
                    elif evalu['position'] == 11:
                        evalu['static'] = "Q7115"
                        blank.append(evalu)
                    elif evalu['position'] == 12:
                        evalu['static'] = "Q7116"
                        blank.append(evalu)
                    elif evalu['position'] == 13:
                        evalu['static'] = "Q7117"
                        blank.append(evalu)
                    elif evalu['position'] == 14:
                        evalu['static'] = "Q7124"
                        blank.append(evalu)
                    elif evalu['position'] == 15:
                        evalu['static'] = "Q7125"
                        blank.append(evalu)

                elif str(dataset['jig_details']['oem_number']) == "0P3648-HS7101":
                    if evalu['position'] == 1:
                        evalu['static'] = "Q7123"
                        blank.append(evalu)
                    elif evalu['position'] == 2:
                        evalu['static'] = "Q7122"
                        blank.append(evalu)
                    elif evalu['position'] == 3:
                        evalu['static'] = "Q7120"
                        blank.append(evalu)
                    elif evalu['position'] == 4:
                        evalu['static'] = "Q7119"
                        blank.append(evalu)
                    elif evalu['position'] == 5:
                        evalu['static'] = "Q7118"
                        blank.append(evalu)
                    elif evalu['position'] == 6:
                        evalu['static'] = "Q7113"
                        blank.append(evalu)
                    elif evalu['position'] == 7:
                        evalu['static'] = "D7129"
                        blank.append(evalu)
                    elif evalu['position'] == 8:
                        evalu['static'] = "Q7112"
                        blank.append(evalu)
                    elif evalu['position'] == 9:
                        evalu['static'] = "D7128"
                        blank.append(evalu)
                    elif evalu['position'] == 10:
                        evalu['static'] = "Q7111"
                        blank.append(evalu)
                    elif evalu['position'] == 11:
                        evalu['static'] = "SW7101"
                        blank.append(evalu)
                    elif evalu['position'] == 12:
                        evalu['static'] = "D7122"
                        blank.append(evalu)
                    elif evalu['position'] == 13:
                        evalu['static'] = "Q7102"
                        blank.append(evalu)
                    elif evalu['position'] == 14:
                        evalu['static'] = "D7125"
                        blank.append(evalu)
                    elif evalu['position'] == 15:
                        evalu['static'] = "Q7101"
                        blank.append(evalu)

                elif str(dataset['jig_details']['oem_number']) == "0P3647-HS7000":
                    if evalu['position'] == 1:
                        evalu['static'] = "Q7023"
                        blank.append(evalu)
                    elif evalu['position'] == 2:
                        evalu['static'] = "Q7022"
                        blank.append(evalu)
                    elif evalu['position'] == 3:
                        evalu['static'] = "Q7021"
                        blank.append(evalu)
                    elif evalu['position'] == 4:
                        evalu['static'] = "Q7015"
                        blank.append(evalu)
                    elif evalu['position'] == 5:
                        evalu['static'] = "Q7014"
                        blank.append(evalu)
                    elif evalu['position'] == 6:
                        evalu['static'] = "Q7013"
                        blank.append(evalu)
                    elif evalu['position'] == 7:
                        evalu['static'] = "SW7001"
                        blank.append(evalu)
                    elif evalu['position'] == 8:
                        evalu['static'] = "07005"
                        blank.append(evalu)
                    elif evalu['position'] == 9:
                        evalu['static'] = "07004"
                        blank.append(evalu)
                    elif evalu['position'] == 10:
                        evalu['static'] = "07003"
                        blank.append(evalu)
                    elif evalu['position'] == 11:
                        evalu['static'] = "07031"
                        blank.append(evalu)
                    elif evalu['position'] == 12:
                        evalu['static'] = "Q7030"
                        blank.append(evalu)
                    elif evalu['position'] == 13:
                        evalu['static'] = "Q7029"
                        blank.append(evalu)
                    elif evalu['position'] == 14:
                        evalu['static'] = "Q7032"
                        blank.append(evalu)
                    elif evalu['position'] == 15:
                        evalu['static'] = "Q7033"
                        blank.append(evalu)


     
		
		            
		    
        print(blank)
        if len(blank) == 0:
            samp['evaluation_data'] = details
        else:
            samp['evaluation_data'] = blank
        samp['status'] = result
        var = str(inspection_id) + "_retry_array"

        retry_array = rch.get_json(var)
        
        if retry_array is None:
            print("length of retry array is :0")
        else:
            print("length of retry array is :"+ str(len(retry_array)))
        
         

        if retry_array is None:
            samp['retry_array'] = []
        else:
            samp['retry_array'] = retry_array
        

        report = {}
        ## cycle time 
        cycle_time_key = str(inspection_id) + "_cycletime"
        curr_cycle_time = rch.get_json(cycle_time_key)
        report['previous_cycle_time'] = str(curr_cycle_time)
        #qty_built : total amt of gvm and gvx built for today 
        Qty_built = 0


        p = [p for p in mp.find().sort( "$natural", -1 )]
        pq = p = [p for p in mp.find()]
        total_inspections = 0 
        total_manual_pass = 0
        for i in p:

            # check for the day wise report 

            start_time = str(i['createdAt'].split(" ")[0])
            current_date = str(datetime.datetime.utcnow().strftime("%Y-%m-%d"))

            if start_time == current_date:

                #break when manual_reset is hit (is_admin_report_reset)

                is_compleated =  i['is_compleated']
                if is_compleated is True:
                    Qty_built = Qty_built + 1

                total_inspections = total_inspections + 1

                if i['is_manual_pass'] is True:
                    total_manual_pass = total_manual_pass + 1

                if i['is_admin_report_reset'] is True:
                    break
                
            
            
        
        report['total_parts_scanned'] = str(Qty_built)      
        report['previous_barcode_number'] = "  "
        # Previous barcode scanned 
        if len(pq) > 1:
            report['previous_barcode_number'] = pq[-2]['serial_no']

        ## total inspection per day until reset
        #total_inspections = len(p) - 1
        #pr = [i for i in mp.find({"is_manual_pass" : True})]
        
        total_auto_pass = str(total_inspections - total_manual_pass)

        fpy = 0.0
        if int(total_inspections) > 0:
            fpy = float(total_auto_pass)  / float(total_inspections)
        report['auto_pass'] = total_auto_pass
        report['manual_pass'] = str(total_manual_pass)
        report['fpy'] = fpy
        samp['reports'] = report
        
        try: del dataset,mp,message,details,var,result,report,p,pr
        except : pass
        
        gc.collect()
        
        #print(samp)
        return samp, 200
    #except:
    #    message = "error fetching inspection_id and status from redis"
    #    status_code = 400
    #    return message,status_code

        

def reject_part(data):
    inspection_id =  data['inspection_id']
    if inspection_id is None:
        message = "inspection_id not provided"
        status_code = 400
        return message,status_code

    try:
        mp = MongoHelper().getCollection('INSPECTION')
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code


    try:
        dataset = mp.find_one({'_id' : ObjectId(inspection_id)})
        if dataset is None:
            message = "Inspection not found in inspection collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid inspection ID"
        status_code = 400
        return message,status_code

    
    dataset['is_reject'] = True

    #dataset['is_compleated'] = True

    try:
        mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  dataset})
    except Exception as e:
        message = "error setting ismanualpass"
        status_code = 400
        return message,status_code

    #make loop continue
    try:
        rch = CacheHelper()
        var = str(ObjectId(dataset['_id'])) + "_paused"
        #is_paused = rch.get_json(var)
        rch.set_json({var:False})
    except:
        return "error setting redis",400
    
    message = dataset
    status_code = 200
    return message,status_code



#force admin pass    
def force_admin_pass(data):

    inspection_id =  data['inspection_id']
    if inspection_id is None:
        message = "inspection_id not provided"
        status_code = 400
        return message,status_code

    try:
        mp = MongoHelper().getCollection('INSPECTION')
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code


    try:
        dataset = mp.find_one({'_id' : ObjectId(inspection_id)})
        if dataset is None:
            message = "Inspection not found in inspection collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid inspection ID"
        status_code = 400
        return message,status_code

    
    dataset['is_manual_pass'] = True
    #dataset['is_compleated'] = True

    try:
        mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  dataset})
    except Exception as e:
        message = "error setting ismanualpass"
        status_code = 400
        return message,status_code

    #make loop continue
    try:
        rch = CacheHelper()
        var = str(ObjectId(dataset['_id'])) + "_paused"
        #is_paused = rch.get_json(var)
        rch.set_json({var:False})
    except:
        return "error setting redis",400

    
    message = dataset
    status_code = 200
    return message,status_code




def get_running_process(): #when someone refresh page

    try:
        mp = MongoHelper().getCollection('INSPECTION')
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    p = [p for p in mp.find()]

    IS_COMPLEATED = True
    dummy_coll = None

    for i in p:
        is_compleated =  i['is_compleated']

        if is_compleated is False:

            IS_COMPLEATED = False

            dummy_coll = i
            break

    if IS_COMPLEATED is False:
        return dummy_coll,200
    else:
        return {},200



def get_process_retry(inspection_id):
    #inspection_id =  data['inspection_id']
    if inspection_id is None:
        message = "inspection_id not provided"
        status_code = 400
        return message,status_code

    try:
        mp = MongoHelper().getCollection('INSPECTION')
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    try:
        dataset = mp.find_one({'_id' : ObjectId(inspection_id)})
        if dataset is None:
            message = "Inspection not found in inspection collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid inspection ID"
        status_code = 400
        return message,status_code

    try:
        rch = CacheHelper()
        var = str(inspection_id) + "_retry_array"

        retry_array = rch.get_json(var)

        if retry_array is None:
            return [],200
        else:
            return retry_array,200
    except:
        
        return "error fetching retry array",400


def continue_process(data):

    inspection_id =  data['inspection_id']
    if inspection_id is None:
        message = "inspection_id not provided"
        status_code = 400
        return message,status_code

    try:
        mp = MongoHelper().getCollection('INSPECTION')
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    try:
        dataset = mp.find_one({'_id' : ObjectId(inspection_id)})
        if dataset is None:
            message = "Inspection not found in inspection collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid inspection ID"
        status_code = 400
        return message,status_code

    try:
        rch = CacheHelper()
        var = str(ObjectId(dataset['_id'])) + "_paused"
        #is_paused = rch.get_json(var)
        rch.set_json({var:False})
    except:
        return "error setting redis",400

    return "process continued",200



