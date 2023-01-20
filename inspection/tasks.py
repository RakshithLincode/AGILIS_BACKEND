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

import uuid
def get_pred_tf_serving_flask(image,port):
    f_name = "/home/schneider/Music/pred_crops/" + str(uuid.uuid4()) + ".png"
    cv2.imwrite(f_name,image)
    #image_expanded = np.expand_dims(image, axis=0)
    data = json.dumps({ 
    #"instances": image_expanded.tolist()
    "instances": f_name
    })
    
    json_data = json.loads(data)

    SERVER_URL = 'http://127.0.0.1:'+str(port)+'/predict'
    
    print(SERVER_URL)
    #print(data)
    response = requests.post(SERVER_URL, json=json_data)
    response.raise_for_status()
    
    #print(response.json())
    prediction = response.json()['prediction']
    
    return prediction
    
def get_pred_tf_serving_flask_black(image,port):
    f_name = "/home/schneider/Music/pred_crops_black/" + str(uuid.uuid4()) + ".png"
    cv2.imwrite(f_name,image)
    #image_expanded = np.expand_dims(image, axis=0)
    data = json.dumps({ 
    #"instances": image_expanded.tolist()
    "instances": f_name
    })
    
    json_data = json.loads(data)

    SERVER_URL = 'http://127.0.0.1:'+str(port)+'/predict'
    
    print(SERVER_URL)
    #print(data)
    response = requests.post(SERVER_URL, json=json_data)
    response.raise_for_status()
    
    #print(response.json())
    prediction = response.json()['prediction']
    
    if prediction is None or prediction == "None":
        return "no_line"
    else:
        return "line"
    

    


"""
def get_pred_tf_serving(image,port,model_type,NUM_CLASSES,PATH_TO_LABELS):
    NUM_CLASSES = int(NUM_CLASSES)
    

    image_expanded = np.expand_dims(image, axis=0)
    accuracy_threshold = 0.7
    #gpu_fraction = 0.4


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
"""






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
    for camera_info in workstation_info['camera_config']['cameras']:
        url = "http://127.0.0.1:8000/livis/v1/preprocess/stream/{}/{}/".format(workstation_id,camera_info['camera_id'])
        # url = "http://0.0.0.0:8000/livis/v1/toyoda/stream1/{}/{}/".format(workstation_id,camera_info['camera_id'])
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
    # role_name = user_details['role_name']
    # print("role_name::::cccccccccccccccccccccccccccccccccccccccccccccccccc",user_details['role_name'])
    user = { "user_id": user_id,
                "role": 'operator',
                "name": ('operatorsecond')
                # "name": (user_details['first_name']+" "+user_details['last_name'])
            }
    #print("user:::: ",user)
    createdAt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
    mp = MongoHelper().getCollection(COLL_NAME)

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
        'is_reject':True,
        'is_compleated' : False,
        'serial_no' : barcode_id,
        'is_admin_report_reset':False,
        'status_end':"",
        'num_retry':None
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
    rch.set_json({"plc_insp_started":True})

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

    all_labelmap_pth = s['all_labelmap_pth']
    all_num_classes = s['all_num_classes']
    all_saved_model_pth = s['all_saved_model_pth']

    black_labelmap_pth = s['black_line_labelmap_pth']
    #black_num_classes = s['black_line_num_classes']
    black_saved_model_pth = s['black_line_saved_model_pth']



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
    copy_of_vendor_match = vendor_match
    #print("vendor match is")
    #print(vendor_match)


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
        #print("999999999999")
        #print(path.exists('/critical_data/regions/'+str(ObjectId(dataset['_id'])) + "_full_img"+".json"))
        if path.exists('/critical_data/regions/'+str(ObjectId(dataset['_id'])) + "_full_img"+".json"):
            f = open ('/critical_data/regions/'+str(ObjectId(dataset['_id'])) + "_full_img"+".json", "r")
            a = json.loads(f.read())
            full_img = a['full_img']
            #print("GOTTTTTTTTTTTTTTTTTTT")
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

    






    def send_crop_pred(j,next_region,width,height,frame,port):

        x_1 = float(j["x"])
        y_1 = float(j["y"])
        w_1 = float(j["w"])
        h_1 = float(j["h"])

        x0_1 = int(x_1 * width)
        y0_1 = int(y_1 * height)
        x1_1 = int(((x_1+w_1) * width))
        y1_1 = int(((y_1+h_1) * height))

        x_2 = float(next_region["x"])
        y_2 = float(next_region["y"])
        w_2 = float(next_region["w"])
        h_2 = float(next_region["h"])

        x0_2 = int(x_2 * width)
        y0_2 = int(y_2 * height)
        x1_2 = int(((x_2+w_2) * width))
        y1_2 = int(((y_2+h_2) * height))

        mid_crp = frame[y0_1:y1_1 , x1_1:x0_2].copy()
        #print("midcrop shape is")
        #print(mid_crp.shape)
        import uuid
        def resize_crop_pred(img):
            #scale_percent = 40 # percent of original size
            width = 720
            height = 1280
            dim = (width, height) 
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC) 
            return resized
            
        f_name = "/home/schneider/Music/tp/" + str(uuid.uuid4()) + ".jpg"
        mid_crp = resize_crop_pred(mid_crp)
        #cv2.imwrite(f_name,mid_crp)
        


        print("going into LINE PREDICTIONS")

        ret = get_pred_tf_serving_flask_black(mid_crp,port)

        return ret
        #if len(objects) == 0:
        #    #no predictions
        #    return "no_line"
        #else:
        #    some predictions -- gotta be line, theres only one label
        #    return "line"





    def black_check(regions,rch,r_key,port,regiona,regionb,regionc,regiond):

        frame  = rch.get_json(r_key)
        height,width,c = frame.shape
        
        #print(regions)

        pred1 = None
        pred2 = None
        pred3 = None

        for j in regions:
            print("reGIONSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
            print(j["cls"])
            print("regionabc")
            print(regiona)
            print(regionb)
            print(regionc)
            

            if regiond is not None:

                if j["cls"] == regiona:
                    print("went in pred1 funct")
                    pred1 = send_crop_pred(j,regions[1],width,height,frame,port)

                elif j["cls"] == regionb:
                    print("went in pred2 funct")
                    pred2 = send_crop_pred(j,regions[2],width,height,frame,port)

                elif j["cls"] == regionc:
                    print("went in pred3 funct")
                    pred3 = send_crop_pred(j,regions[3],width,height,frame,port)

                

            else:

                if j["cls"] == regiona:
                    pred1 = send_crop_pred(j,regions[1],width,height,frame,port)

                elif j["cls"] == regionb:
                    pred2 = send_crop_pred(j,regions[2],width,height,frame,port)

                #elif j["cls"] == regionc and j+1["cls"] == regiond:
                #    pred3 = send_crop_pred(j,width,height,frame,port)

        if regiond is not None:
            return pred1,pred2,pred3
        else:
            return pred1,pred2
        



    def regions_crop_pred(regions,rch,r_key,final_dct,port):
        global frame
        global crp
        t1 = time.time()
        frame  = rch.get_json(r_key)
        t2 = time.time()
        #print('time taken to get frame from redis  ::  ::  ::  :: ---------    '+ str(t2-t1))

        height,width,c = frame.shape
        print(frame.shape)
        #height = height*3
        #width = width*3

        def resize_crop(img):
            scale_percent = 40 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height) 
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS64) 
            return resized
            
        def resize_crop_pred(img):
            #scale_percent = 40 # percent of original size
            width = 720
            height = 1280
            dim = (width, height) 
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC) 
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
            
            import uuid
        
            f_name = "/home/schneider/Music/full/" + str(uuid.uuid4()) + ".jpg"
            
            crp = resize_crop_pred(crp)
            #cv2.imwrite(f_name,crp)
            t2 = time.time()
            #print('time taken to get crop            ::  ::  ::  :: ---------    '+ str(t2-t1))
            
            
            #cv2.imwrite('/critical_data/tmpcrops/'+unique_id+'.jpg',crp)
            #print('/critical_data/tmpcrops/'+unique_id+'.jpg')

            print("going into prediction")
            print(crp.shape)
            t1 = time.time()
            predicted_obj = get_pred_tf_serving_flask(crp,port)
            final_dct[label] = predicted_obj
                
        return final_dct




    #write a while true loop : if final dict match with kanban  or manual pass by admin using inspection_id
    t_start_inspection = time.time()

    loop_idx_for_del = 0

    def do_del(cl_obj):
        try:
            del cl_obj
        except Exception as e:
            pass
            #print("del ops")
            #print(e)
    retry_list = []




    while(True):
        if retry_list != []:

            var = str(ObjectId(dataset['_id'])) + "_paused"
            is_paused = rch.get_json(var)
            #rch.set_json({var:p1_lst})

            if is_paused:
                #print("i am paused")
                continue
                


        #print("Testing here")
        try:
            COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
            mp = MongoHelper().getCollection(COLL_NAME)
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
        print(full_img)
        #print(cam_list)
        #print(key_list)

                  
        def get_pred_left_camera(p2_lst,jig_type):
            for cam,r_key in zip(cam_list,key_list):
                if cam == "left_camera":
                    for f in full_img:
                        if f['cam_name'] == 'left_camera':
                            try:
                                regions = f['regions']
                                if regions != "":
                                    print("inside left cam")
                                    p2_lst = regions_crop_pred(regions,rch,r_key,p2_lst,5001)
                                    if jig_type == "GVX":
                                        p4,p5,p6 = black_check(regions,rch,r_key,5006,"region5","region6","region7","region8")
                                        print("FUNCTION CALLED LEFT CAM")
                                        var1 = str(inspection_id) + "_line4"
                                        var2 = str(inspection_id) + "_line5"
                                        var3 = str(inspection_id) + "_line6"
                                        rch.set_json({var1:p4})
                                        rch.set_json({var2:p5})
                                        rch.set_json({var3:p6})

                                    var = str(inspection_id) + "_cam2"
                                    rch.set_json({var:p2_lst})
                            except Exception as e:
                                print("region not defined in left camera:" + str(e) )
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

                                    p4_lst = regions_crop_pred(regions,rch,r_key,p4_lst,5003)
                                    if jig_type == "GVX":
                                        p1,p2,p3 = black_check(regions,rch,r_key,5008,"region13","region14","region15","region16")
                                        print("FUNCTION CALLED right CAM")
                                        var1 = str(inspection_id) + "_line10"
                                        var2 = str(inspection_id) + "_line11"
                                        var3 = str(inspection_id) + "_line12"
                                        rch.set_json({var1:p1})
                                        rch.set_json({var2:p2})
                                        rch.set_json({var3:p3})
                                    var = str(inspection_id) + "_cam4"
                                    rch.set_json({var:p4_lst})
                            except Exception as e:
                                print("region not defined in right camera:" + str(e) )
                                pass  

        t1 = time.time()
        #check if something is running - if yes check what it is ? if selected and running match then pass else kill running and start selected


        

        def launch_all_containers():
            t1 = time.time()
            
            print("starting all flask supervisorssss")
            cmd1 = "supervisorctl start flask_service_5001"
            cmd3 = "supervisorctl start flask_service_5003"
                    
            cmd6 = "supervisorctl start flask_service_5006"
            cmd8 = "supervisorctl start flask_service_5008"

                    
            subprocess.Popen(cmd1,shell=True).wait()
            subprocess.Popen(cmd3,shell=True).wait()
                    
            subprocess.Popen(cmd6,shell=True).wait()
            subprocess.Popen(cmd8,shell=True).wait()
            print("all flask started...")
        

            print("all supervisors launched")
            t2 = time.time()
            print('all supervisor making it up and running time is            ::  ::  ::  :: ---------    '+ str(t2-t1))

        #proc = subprocess.run(['docker','container','ls'],check=True,stdout=PIPE)
        proc1 = subprocess.run(['supervisorctl','status','flask_service_5001'],stdout=PIPE)
        proc3 = subprocess.run(['supervisorctl','status','flask_service_5003'],stdout=PIPE)
        
        proc6 = subprocess.run(['supervisorctl','status','flask_service_5006'],stdout=PIPE)
        proc8 = subprocess.run(['supervisorctl','status','flask_service_5008'],stdout=PIPE)
        
        HAS_RUN = True
        
        
        if 'RUNNING' in str(proc1):
            pass
        else:
            HAS_RUN = False
        
    
        
        if 'RUNNING' in str(proc3):
            pass
        else:
            HAS_RUN = False
        
            
        if 'RUNNING' in str(proc6):
            pass
        else:
            HAS_RUN = False
            
    
        if 'RUNNING' in str(proc8):
            pass
        else:
            HAS_RUN = False
        
        
      

        if HAS_RUN is True:
            pass
        else:
            #no containers running # launch appropriate container
            print("some supervisors are not running")
            launch_all_containers()  


        weights_folder1 = "/home/schneider/Music/pred_crops"
        if not os.path.exists(weights_folder1):
            os.makedirs(weights_folder1)
        else:
            shutil.rmtree(weights_folder1, ignore_errors=True)
            os.makedirs(weights_folder1)

        weights_folder1 = "/home/schneider/Music/pred_crops_black"
        if not os.path.exists(weights_folder1):
            os.makedirs(weights_folder1)
        else:
            shutil.rmtree(weights_folder1, ignore_errors=True)
            os.makedirs(weights_folder1)
            
               
        p1_lst = {}
        p2_lst = {}
        p3_lst = {}
        p4_lst = {}
        p5_lst = {}
        
        var = str(inspection_id) + "_cam1"
        rch.set_json({var:p1_lst})
        var = str(inspection_id) + "_cam2"
        rch.set_json({var:p2_lst})
        
        P1 = Process(target=get_pred_left_camera,args=(p2_lst,jig_type,))
        P2 = Process(target=get_pred_right_camera,args=(p4_lst,jig_type,))
        
        P1.start()
        P2.start()
        P1.join()
        P2.join()
       
        
        
        t2 = time.time()
        print('TIMEEEEEEEEEEEEEEEEEEEEEEEEEEE            ::  ::  ::  :: ---------    '+ str(t2-t1))
        var = str(inspection_id) + "_prediction_time"
        rch.set_json({var:int(t2-t1)})
        
        
        
        var = str(inspection_id) + "_cam1"
        p1_lst = rch.get_json(var)
        var = str(inspection_id) + "_cam2"
        p2_lst = rch.get_json(var)

        #line logic -- 

        if jig_type == "GVX":

            lst = []
            HAS_LINE = None

            var1 = str(inspection_id) + "_line1"
            res1 = rch.get_json(var1)
            lst.append(res1)
            var1 = str(inspection_id) + "_line2"
            res2 = rch.get_json(var1)
            lst.append(res2)
            var1 = str(inspection_id) + "_line3"
            res3 = rch.get_json(var1)
            lst.append(res3)
            var1 = str(inspection_id) + "_line4"
            res4 = rch.get_json(var1)
            lst.append(res4)
            var1 = str(inspection_id) + "_line5"
            res5 = rch.get_json(var1)
            lst.append(res5)
            var1 = str(inspection_id) + "_line6"
            res6 = rch.get_json(var1)
            lst.append(res6)
            var1 = str(inspection_id) + "_line7"
            res7 = rch.get_json(var1)
            lst.append(res7)
            var1 = str(inspection_id) + "_line8"
            res8 = rch.get_json(var1)
            lst.append(res8)
            var1 = str(inspection_id) + "_line9"
            res9 = rch.get_json(var1)
            lst.append(res9)
            var1 = str(inspection_id) + "_line10"
            res10 = rch.get_json(var1)
            lst.append(res10)
            var1 = str(inspection_id) + "_line11"
            res11 = rch.get_json(var1)
            lst.append(res11)
            var1 = str(inspection_id) + "_line12"
            res12 = rch.get_json(var1)
            lst.append(res12)
            var1 = str(inspection_id) + "_line13"
            res13 = rch.get_json(var1)
            lst.append(res13)
            var1 = str(inspection_id) + "_line14"
            res14 = rch.get_json(var1)
            lst.append(res14)

            freq_black = {} 
            for item in lst: 
                if (item in freq_black): 
                    freq_black[item] += 1
                else: 
                    freq_black[item] = 1

            print("freq_balc")
            print(freq_black)
            try:
                line_f = freq_black['line']
            except:
                line_f = 0
            try:
                no_line_f = freq_black['no_line']
            except:
                no_line_f = 0

            if line_f > no_line_f:
                print("line found")
                #has line
                HAS_LINE = True

            elif line_f < no_line_f:
                print("NO line found")
                #no line
                HAS_LINE = False

            elif line_f == no_line_f:
                #confusion -- make
                print("HAVINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG CONFUSION ON LINEEEEEEEEEEEEEEEEE PREDICTIONNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
                HAS_LINE = True #false predict 

            
        
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #print(p1_lst)
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        final_dct = {}
        final_dct.update(p1_lst)
        final_dct.update(p2_lst)
        final_dct.update(p3_lst)
        final_dct.update(p4_lst)
        final_dct.update(p5_lst)
        
        #weights_folder1 = "/home/schneider/Music/pred_crops"
        #if not os.path.exists(weights_folder1):
        #    os.makedirs(weights_folder1)
        #else:
        #    shutil.rmtree(weights_folder1, ignore_errors=True)
        #    os.makedirs(weights_folder1)


        #write black line det logic here

	
        

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
        print(region_pass_fail)
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
        vendor_match = copy_of_vendor_match
        print(vendor_match)
        for v in vendor_match:
            #v = str(v)
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


      
        if jig_type == "LOADSTAR":
            if HAS_LINE:
                var = str(inspection_id) + "_hasline"
                rch.set_json({var:True})
            else:
                var = str(inspection_id) + "_hasline"
                rch.set_json({var:False})

        #write to redis : key is inspection_id and value is region_pass_fail
        rch.set_json({inspection_id:region_pass_fail})
        var = str(inspection_id) + "_result"
        rch.set_json({var:"fail"})
        #rch.set_json({"plc_insp_status":False})
        
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
    rch.set_json({"plc_insp_status":True})

    try:
        COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
        mp = MongoHelper().getCollection(COLL_NAME)
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

    dataset['completedAt'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    completedAt = datetime.datetime.strptime(dataset['completedAt'], '%Y-%m-%d %H:%M:%S')
    createdAt = datetime.datetime.strptime(dataset['createdAt'], '%Y-%m-%d %H:%M:%S')
    duration = completedAt - createdAt
    dataset['duration'] = str(duration)
    dataset['status'] = 'completed'
    dataset['is_compleated'] = True
    
    manual_pss = dataset['is_manual_pass']
    manual_rej = dataset['is_reject']
    
    if manual_pss:
        dataset['status_end'] = "Manually Pass"
    elif manual_rej:
        dataset['status_end'] = "Manually Rejected"
    else:
        dataset['status_end'] = "Auto Pass"
    
    var = str(inspection_id) + "_retry_array"

    retry_array = rch.get_json(var)

    if retry_array is None:
        dataset['num_retry'] = 0
    else:
        dataset['num_retry'] = int(len(retry_array)) - 1 
    
    
    
    mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  dataset})

    #do gc collect here
    gc.collect()
    
    #restart django server 
    subprocess.Popen("supervisorctl restart django_service",shell=True).wait()




#get method
def get_current_inspection_details_utils(inspection_id):
    #{eval_data}

    #inspection_id =  data['inspection_id']
    if inspection_id is None:
        message = "inspection_id not provided"
        status_code = 400
        return message,status_code

    try:
        COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
        mp = MongoHelper().getCollection(COLL_NAME)
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

    jig_type = dataset['jig_details']['jig_type']
    curr_ser_numb = dataset['serial_no']

    samp = {}
    if True:
        #samp = {}
        rch = CacheHelper()
        details = rch.get_json(str(inspection_id))
        var = str(inspection_id) + "_result"
        result = rch.get_json(var)
        
        if details is not None:
            print("details is not None")
            print(details)
     
            
        samp['has_line'] = False
        if jig_type == "AGILIS":
            var = str(inspection_id) + "_hasline"
            HAS_LINE = rch.get_json(var)

            if HAS_LINE:
                samp['has_line'] = True
                samp['status'] = "fail"
            else:
                samp['status'] = result
        else:
            samp['status'] = result 
            
        
        """
        if len(blank) == 0:
            samp['evaluation_data'] = details
        else:
            samp['evaluation_data'] = blank
        """
        if details is None:
            samp['evaluation_data'] = details
        else:
            a_z = details.copy()
            for i_z in a_z:
                new_part_number = []
                part_number = i_z['part_number']
                for j_z in part_number:
                    if '_' in str(j_z): 
                        s_z = j_z.split('_')[0]
                        new_part_number.append(s_z)
                    else:
                        new_part_number.append(j_z)
                i_z['part_number'] = new_part_number
                new_res_pn = i_z['result_part_number']
                if new_res_pn is not None:
                    if '_' in i_z['result_part_number']:
                        i_z['result_part_number'] = str(new_res_pn).split('_')[0]
                
            samp['evaluation_data'] = a_z
        
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
        report['process_time'] = str(curr_cycle_time)
        var = str(inspection_id) + "_prediction_time"
        prediction_time = rch.get_json(var)
        report['previous_cycle_time'] = str(prediction_time) + " sec"
        #qty_built : total amt of gvm and gvx built for today 
        Qty_built = 0


        p = [p for p in mp.find().sort( "$natural", -1 )]
        pq = [p for p in mp.find().sort( "$natural", -1 )]
        #pq = [p for p in mp.find()]
        
        total_inspections = 0 
        total_manual_pass = 0
        total_multi_try = 0
        total_one_time_pass = 0
        for i in p:

            # check for the day wise report 

            start_time = str(i['createdAt'].split(" ")[0])
            current_date = str(datetime.datetime.now().strftime("%Y-%m-%d"))

            if start_time == current_date:

                #break when manual_reset is hit (is_admin_report_reset)

                is_compleated =  i['is_compleated']
                if is_compleated is True:
                    Qty_built = Qty_built + 1

                total_inspections = total_inspections + 1

                if i['is_manual_pass'] is True:
                    total_manual_pass = total_manual_pass + 1
                    
                num_retry =  i['num_retry']
                if num_retry is not None:
                    if num_retry > 0:
                        total_multi_try = total_multi_try + 1

                if i['is_admin_report_reset'] is True:
                    break
                
        
        report['total_parts_scanned'] = str(Qty_built)      
        report['previous_barcode_number'] = "AGFS123465"
        # Previous barcode scanned 
        if len(pq) >= 2:
            #report['previous_barcode_number'] = pq[1]['serial_no']
            report['previous_barcode_number'] = str(curr_ser_numb)
        ## total inspection per day until reset
        #total_inspections = len(p) - 1
        #pr = [i for i in mp.find({"is_manual_pass" : True})]
        print("total inspections",total_inspections)
        print("total_multi_try",total_multi_try)
        total_auto_pass = str(total_inspections - total_manual_pass)
        total_one_time_pass = str(total_inspections - total_multi_try)

        fpy = 0.0
        if int(total_inspections) > 0:
            fpy = float(total_one_time_pass)  / float(total_inspections)
            fpy = (fpy*100)
            fpy = round(fpy,2)
            fpy = str(fpy) + " %"
        report['auto_pass'] = total_auto_pass
        report['manual_pass'] = str(total_manual_pass)
        report['fpy'] = fpy
        samp['reports'] = report
        
        try: del dataset,mp,message,details,var,result,report,p,pr
        except : pass
        
        gc.collect()
        samp = {"Message": "Success!", "data": {"has_line": True, "status": 'Accepted', "evaluation_data": ['k150','k150','k150','k150','k150','k150','k150','k150','k150','k150'], "retry_array": [], "reports": {"process_time": "2021-03-31 16:12:20", "previous_cycle_time": "2021-03-31 16:12:20", "total_parts_scanned": "15", "previous_barcode_number": "F22112007599", "auto_pass": "1", "manual_pass": "0", "fpy": "100.0 %" , "mes_check":"offline"}}}
        #print(samp)
        return samp, 200
    #except:
    #    message = "error fetching inspection_id and status from redis"
    #    status_code = 400
    #    return message,status_code

        

def reject_part(data):
    inspection_id =  data['inspection_id']
    approved_by = data['admin_id']
    if inspection_id is None:
        message = "inspection_id not provided"
        status_code = 400
        return message,status_code

    try:
        COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
        mp = MongoHelper().getCollection(COLL_NAME)
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
    coll = { 
        'approved_by' : approved_by,
        'is_reject' : True
    }

    #dataset['is_compleated'] = True

    try:
        mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  coll})
    except Exception as e:
        message = "error setting ismanualpass"
        status_code = 400
        return message,status_code

    #make loop continue
    try:
        rch = CacheHelper()
        rch.set_json({"plc_insp_status":False})
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
    approved_by = data['admin_id']
    if inspection_id is None:
        message = "inspection_id not provided"
        status_code = 400
        return message,status_code

    try:
        COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
        mp = MongoHelper().getCollection(COLL_NAME)
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
 
    
    coll = { 
        'approved_by' : approved_by,
        'is_manual_pass' : True
    }

    try:
        mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  coll})
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
        rch.set_json({"plc_insp_status":True})
    except:
        return "error setting redis",400

    
    message = dataset
    status_code = 200
    return message,status_code




def get_running_process(): #when someone refresh page

    try:
        COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
        mp = MongoHelper().getCollection(COLL_NAME)
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
        COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
        mp = MongoHelper().getCollection(COLL_NAME)
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
        COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
        mp = MongoHelper().getCollection(COLL_NAME)
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

def get_static(jig_id):
    #inspection_id =  data['inspection_id']
    #jig_id = data['jig_id']

    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    #jig_id =  data['jig_id']
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
    kanban = dataset['kanban']

    a = {}

    if str(oem_number) == "0P3621-HS7000":
        dct = []
        pos = {'position':1,'static':"Q7018 325-7560-Z"}
        dct.append(pos)
        pos = {'position':2,'static':"Q7019 325-7560-Z"}
        dct.append(pos)
        pos = {'position':3,'static':"Q7022 325-7560-Z"}
        dct.append(pos)
        pos = {'position':4,'static':"Q7023 325-7560-Z"}
        dct.append(pos)
        pos = {'position':5,'static':"Q7012 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':6,'static':"Q7013 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':7,'static':"Q7014 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':8,'static':"Q7015 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':9,'static':"SW7001 541-9037-Z-001"}
        dct.append(pos)
        pos = {'position':10,'static':"Q7004 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':11,'static':"Q7005 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':12,'static':"Q7006 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':13,'static':"Q7007 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':14,'static':"Q7026 325-7560-Z"}
        dct.append(pos)
        pos = {'position':15,'static':"Q7027 325-7560-Z"}
        dct.append(pos)
        pos = {'position':16,'static':"Q7030 325-7560-Z"}
        dct.append(pos)
        pos = {'position':17,'static':"Q7031 325-7560-Z"}
        dct.append(pos)
        pos = {'position':18,'static':"Q7032 325-7560-Z"}
        dct.append(pos)
        pos = {'position':19,'static':"Q7033 325-7560-Z"}
        dct.append(pos)

        

    elif str(oem_number) == "0P3621-HS7001":
        dct = []
        pos = {'position':1,'static':"Q7034 325-7560-Z"}
        dct.append(pos)
        pos = {'position':2,'static':"Q7035 325-7560-Z"}
        dct.append(pos)
        pos = {'position':3,'static':"Q7028 325-7560-Z"}
        dct.append(pos)
        pos = {'position':4,'static':"Q7029 325-7560-Z"}
        dct.append(pos)
        pos = {'position':5,'static':"Q7024 325-7560-Z"}
        dct.append(pos)
        pos = {'position':6,'static':"Q7025 325-7560-Z"}
        dct.append(pos)
        pos = {'position':7,'static':"Q7000 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':8,'static':"Q7001 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':9,'static':"Q7002 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':10,'static':"Q7003 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':11,'static':"SW7000 541-9037-Z-001"}
        dct.append(pos)
        pos = {'position':12,'static':"Q7008 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':13,'static':"Q7009 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':14,'static':"Q7010 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':15,'static':"Q7011 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':16,'static':"Q7020 325-7560-Z"}
        dct.append(pos)
        pos = {'position':17,'static':"Q7021 325-7560-Z"}
        dct.append(pos)
        pos = {'position':18,'static':"Q7016 325-7560-Z"}
        dct.append(pos)
        pos = {'position':19,'static':"Q7017 325-7560-Z"}
        dct.append(pos)

        


    elif str(oem_number) == "0P3623-HS7100":
        dct = []
        pos = {'position':1,'static':"Q7122 325-7560-Z"}
        dct.append(pos)
        pos = {'position':2,'static':"Q7123 325-7560-Z"}
        dct.append(pos)
        pos = {'position':3,'static':"Q7118 325-7560-Z"}
        dct.append(pos)
        pos = {'position':4,'static':"Q7119 325-7560-Z"}
        dct.append(pos)
        pos = {'position':5,'static':"Q7120 325-7560-Z"}
        dct.append(pos)
        pos = {'position':6,'static':"Q7121 325-7560-Z"}
        dct.append(pos)
        pos = {'position':7,'static':"Q7108 325-4760-Z"}
        dct.append(pos)
        pos = {'position':8,'static':"D7127 340-6061-Z"}
        dct.append(pos)
        pos = {'position':9,'static':"Q7109 325-4760-Z"}
        dct.append(pos)
        pos = {'position':10,'static':"SW7101 541-0076-Z"}
        dct.append(pos)
        pos = {'position':11,'static':"Q7112 325-4760-Z"}
        dct.append(pos)
        pos = {'position':12,'static':"D7129 340-6061-Z"}
        dct.append(pos)
        pos = {'position':13,'static':"Q7113 325-4760-Z"}
        dct.append(pos)
        pos = {'position':14,'static':"D7125 340-1240-Z"}
        dct.append(pos)
        pos = {'position':15,'static':"Q7100 325-0148-Z"}
        dct.append(pos)
        pos = {'position':16,'static':"Q7101 325-0148-Z"}
        dct.append(pos)
        pos = {'position':17,'static':"Q7102 325-0148-Z"}
        dct.append(pos)
        pos = {'position':18,'static':"D7122 340-1230-Z"}
        dct.append(pos)
        pos = {'position':19,'static':"D7123 340-1230-Z"}
        dct.append(pos)

        

    

    elif str(oem_number) == "0P3623-HS7101":
        dct = []
        pos = {'position':1,'static':"D7120 340-1230-Z"}
        dct.append(pos)
        pos = {'position':2,'static':"D7121 340-1230-Z"}
        dct.append(pos)
        pos = {'position':3,'static':"Q7105 325-0148-Z"}
        dct.append(pos)
        pos = {'position':4,'static':"Q7104 325-0148-Z"}
        dct.append(pos)
        pos = {'position':5,'static':"Q7103 325-0148-Z"}
        dct.append(pos)
        pos = {'position':6,'static':"D7124 340-1240-Z"}
        dct.append(pos)
        pos = {'position':7,'static':"Q7111 325-4760-Z"}
        dct.append(pos)
        pos = {'position':8,'static':"D7128 340-6061-Z"}
        dct.append(pos)
        pos = {'position':9,'static':"Q7110 325-4760-Z"}
        dct.append(pos)
        pos = {'position':10,'static':"SW7100 541-0076-Z"}
        dct.append(pos)
        pos = {'position':11,'static':"Q7107 325-4760-Z"}
        dct.append(pos)
        pos = {'position':12,'static':"D7126 340-6061-Z"}
        dct.append(pos)
        pos = {'position':13,'static':"Q7106 325-4760-Z"}
        dct.append(pos)
        pos = {'position':14,'static':"Q7115 325-7560-Z"}
        dct.append(pos)
        pos = {'position':15,'static':"Q7114 325-7560-Z"}
        dct.append(pos)
        pos = {'position':16,'static':"Q7117 325-7560-Z"}
        dct.append(pos)
        pos = {'position':17,'static':"Q7118 325-7560-Z"}
        dct.append(pos)
        pos = {'position':18,'static':"Q7125 325-7560-Z"}
        dct.append(pos)
        pos = {'position':19,'static':"Q7124 325-7560-Z"}
        dct.append(pos)

        

    elif str(oem_number) == "0P3647-HS7001":
        dct = []
        pos = {'position':1,'static':"Q7034 325-5060-Z"}
        dct.append(pos)
        pos = {'position':2,'static':"Q7035 325-5060-Z"}
        dct.append(pos)
        pos = {'position':3,'static':"Q7025 325-7560-Z"}
        dct.append(pos)
        pos = {'position':4,'static':"Q7026 325-7560-Z"}
        dct.append(pos)
        pos = {'position':5,'static':"Q7027 325-7560-Z"}
        dct.append(pos)
        pos = {'position':6,'static':"Q7000 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':7,'static':"Q7001 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':8,'static':"Q7002 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':9,'static':"SW7000 541-0076-Z-001"}
        dct.append(pos)
        pos = {'position':10,'static':"Q7010 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':11,'static':"Q7011 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':12,'static':"Q7012 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':13,'static':"Q7017 325-7560-Z"}
        dct.append(pos)
        pos = {'position':14,'static':"Q7019 325-7560-Z"}
        dct.append(pos)
        pos = {'position':15,'static':"Q7019 325-7560-Z"}
        dct.append(pos)

        

    elif str(oem_number) == "0P3648-HS7100":
        dct = []
        pos = {'position':1,'static':"Q7103 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':2,'static':"D7124 340-1240-Z"}
        dct.append(pos)
        pos = {'position':3,'static':"Q7104 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':4,'static':"D7120 HUA28208"}
        dct.append(pos)
        pos = {'position':5,'static':"SW7100 541-0076-Z-001"}
        dct.append(pos)
        pos = {'position':6,'static':"Q7107 325-4760-Z"}
        dct.append(pos)
        pos = {'position':7,'static':"D7126 340-6260-Z"}
        dct.append(pos)
        pos = {'position':8,'static':"Q7108 325-4760-Z"}
        dct.append(pos)
        pos = {'position':9,'static':"D7127 340-6260-Z"}
        dct.append(pos)
        pos = {'position':10,'static':"Q7109 325-4760-Z"}
        dct.append(pos)
        pos = {'position':11,'static':"Q7115 325-7560-Z"}
        dct.append(pos)
        pos = {'position':12,'static':"Q7116 325-5060-Z"}
        dct.append(pos)
        pos = {'position':13,'static':"Q7117 325-5060-Z"}
        dct.append(pos)
        pos = {'position':14,'static':"Q7124 325-5060-Z"}
        dct.append(pos)
        pos = {'position':15,'static':"Q7125 325-5060-Z"}
        dct.append(pos)


    elif str(oem_number) == "0P3648-HS7101":
        dct = []
        pos = {'position':1,'static':"Q7123 325-5060-Z"}
        dct.append(pos)
        pos = {'position':2,'static':"Q7122 325-5060-Z"}
        dct.append(pos)
        pos = {'position':3,'static':"Q7120 325-7560-Z"}
        dct.append(pos)
        pos = {'position':4,'static':"Q7119 325-5060-Z"}
        dct.append(pos)
        pos = {'position':5,'static':"Q7118 325-5060-Z"}
        dct.append(pos)
        pos = {'position':6,'static':"Q7113 325-4760-Z"}
        dct.append(pos)
        pos = {'position':7,'static':"D7129 340-6260-Z"}
        dct.append(pos)
        pos = {'position':8,'static':"Q7112 325-4760-Z"}
        dct.append(pos)
        pos = {'position':9,'static':"D7128 340-6260-Z"}
        dct.append(pos)
        pos = {'position':10,'static':"Q7111 325-4760-Z"}
        dct.append(pos)
        pos = {'position':11,'static':"SW7101 541-0076-Z-001"}
        dct.append(pos)
        pos = {'position':12,'static':"D7122 HUA28208"}
        dct.append(pos)
        pos = {'position':13,'static':"Q7102 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':14,'static':"D7125 340-1240-Z"}
        dct.append(pos)
        pos = {'position':15,'static':"Q7101 325-0148-Z-001"}
        dct.append(pos)


    elif str(oem_number) == "0P3647-HS7000":
        dct = []
        pos = {'position':1,'static':"Q7023 325-7560-Z"}
        dct.append(pos)
        pos = {'position':2,'static':"Q7022 325-7560-Z"}
        dct.append(pos)
        pos = {'position':3,'static':"Q7021 325-7560-Z"}
        dct.append(pos)
        pos = {'position':4,'static':"Q7015 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':5,'static':"Q7014 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':6,'static':"Q7013 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':7,'static':"SW7001 541-0076-Z-001"}
        dct.append(pos)
        pos = {'position':8,'static':"07005 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':9,'static':"07004 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':10,'static':"07003 325-0148-Z-001"}
        dct.append(pos)
        pos = {'position':11,'static':"07031 325-7560-Z"}
        dct.append(pos)
        pos = {'position':12,'static':"Q7030 325-7560-Z"}
        dct.append(pos)
        pos = {'position':13,'static':"Q7029 325-7560-Z"}
        dct.append(pos)
        pos = {'position':14,'static':"Q7032 325-5060-Z"}
        dct.append(pos)
        pos = {'position':15,'static':"Q7033 325-5060-Z"}
        dct.append(pos)
    else:
        dct = []
    """
    b = []
    print("dct is")
    print(dct)
    for k,i in zip(kanban,dct):
        if i['position'] == k['position']:
            a['position'] = i['position']
            a['static'] = i['static']
            a['part_number'] = k['part_number']

            b.append(a)
    
    for k in kanban:
        for i in dct:
        

            if i['position'] == k['position']:
                a['position'] = i['position']
                a['static'] = i['static']
                a['part_number'] = k['part_number']

                b.append(a)
    """        
    #print(b)
    return dct,200




def admin_report_reset(data):

    #inspection_id = data["inspection_id"]
    approved_by = data["admin_id"]



    try:
        mp = MongoHelper().getCollection("INSPECTION")
    except Exception as e:
        message = "Cannot connect to db"
        return message, status_code

    p = [p for p in mp.find().sort( "$natural", -1 )]
    
    
    p = p[0]
    

    coll = {"reset_by": approved_by, "is_admin_report_reset": True}

    try:
        mp.update({"_id": ObjectId(p["_id"])}, {"$set": coll})
    except Exception as e:
        message = "error setting reset report"
        status_code = 400
        return message, status_code

    return p, 200

