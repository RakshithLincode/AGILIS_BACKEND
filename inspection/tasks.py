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
#from livis.Monk_Object_Detection.tf_obj_2.lib.models.research.object_detection.webcam import load_model_to_memory,crop_infer
from livis.models.research.object_detection.utils import label_map_util
import os
# %matplotlib inline
import cv2
import pandas as pd
from pymongo import MongoClient
from livis import settings as s
from iteration_utilities import unique_everseen
import concurrent.futures
import urllib.request
import ast
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
        'is_reject':False,
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
            COL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
            mp = MongoHelper().getCollection(COL_NAME)
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
        
        def parallel_process():
            URLS = ['http://35.193.71.60:5000/predict',
                    'http://35.193.71.60:5001/predict']
            def load_url(url, timeout):
                with urllib.request.urlopen(url, timeout=timeout) as conn:
                    return conn.read()
            value  = []
            with concurrent.futures.ThreadPoolExecutor(4) as executor:
                future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    data = future.result()
                    value.append(data)
                return value  

        def convert_value(value):
            dict = {'prediction':[]}
            final_dict = {}
            prediction_1 = ast.literal_eval(value[0].decode('utf-8'))
            prediction_2 =  ast.literal_eval(value[1].decode('utf-8'))
            for i in prediction_2['prediction']:
                dict['prediction'].append(i)
            for i in prediction_1['prediction']:
                dict['prediction'].append(i) 
            list_dict = sorted(dict['prediction'], key=lambda d: d['position'])
            final_dict['prediction'] = list_dict 
            return final_dict  

        def find_empty_value(dict):
            result_prediction = {'prediction':[]}
            for i in dict['prediction']:
                if '' in i['part_number']:
                    pass
                else:
                    result_prediction['prediction'].append({"part_number":i['part_number'],"position":i['position']})
            return result_prediction
                
        def get_kanban(oem_number):
            mp = MongoHelper().getCollection("jig")
            for x in mp.find():
                if x['oem_number'] == oem_number:
                    kanban = x.get('kanban')	
                    return kanban

        def check_kanban(actual_value,predicted_value):
            position = {'position_present':[],'position_absent':[]}
            position_actual = []
            position_predict = []
            isAccepted  = []
            status = []
            for actual,predicted in zip(actual_value,predicted_value):
                if actual['part_number'] == predicted['part_number']:
                    position['position_present'].append(actual['position'])
                    isAccepted.append(True)  
                else:
                    position['position_absent'].append(actual['position'])
                    isAccepted.append(False)  
            for value in actual_value:
                position_actual.append(value['position'])
            for value in predicted_value:
                position_predict.append(value['position']) 
            for i in position_actual:
                if i in position_predict:
                    pass
                else:
                    position['position_absent'].append(i)
                    isAccepted.append(False)  
            if False in isAccepted:
                position['status'] = False
            else:
                position['status'] = True
            return  position , isAccepted 

        def find_region_pass_fail(predicted_value,each_value_status,actual_value):
            for i,j,k in zip(predicted_value,each_value_status,actual_value):
                i['result_part_number'] = i['part_number']
                del i["part_number"]
                i['status']=j
                if j == 'Accepted':
                    i['color']='green'
                else:
                    i['color']='red'
                i['part_number'] =  k['part_number'] 
            return predicted_value 

        
        final_dct = {}
        print(oem_number,'hjoemmmmmmmmmmmmmmmmmmmmmmmmm')
        actual_value = get_kanban(str(oem_number))
        print(actual_value,'kkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
        predict_ocr_value = parallel_process()
        result_dict = convert_value(predict_ocr_value)
        result_prediction = find_empty_value(result_dict)
        predicted_value = list(unique_everseen(result_prediction['prediction'])) 
        position,each_value_status  = check_kanban(actual_value,predicted_value)
        region_pass_fail = find_region_pass_fail(predicted_value,each_value_status,actual_value)
        print(region_pass_fail,'jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj')
        print(position)
        t2 = time.time()
        HAS_LINE = True #false predict 
        if jig_type == "GVX":
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
        print(IS_PROCESS_END,'isprocesssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
        if IS_PROCESS_END is True:
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
        try: do_del(final_dct)
        except: pass  
        try: do_del(var)
        except: pass
        try: do_del(key)
        except: pass
        try: do_del(vendor_match)
        except: pass

        try: gc.collect()
        except: pass        
        
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


def get_current_inspection_details_utils(inspection_id):
    #{eval_data}
    # inspection_id = '63dfce54d4a606636992653e'

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
        print(details,'detailsssssssssssssssssssssssssssssssssss')
        var = str(inspection_id) + "_result"
        result = rch.get_json(var)
        
        if details is not None:
            print("details is not None")
            print(details)
     
        samp['status'] = result 

        blank = []
        """
        if len(blank) == 0:
            samp['evaluation_data'] = details
        else:
            samp['evaluation_data'] = blank
        """
        if details is None:
            samp['evaluation_data'] = details
        else:
        
            if len(blank) == 0:
        
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
            else:
                a_z = blank.copy()
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
        report['previous_barcode_number'] = "  "
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
        
        #print(samp)
        return samp, 200
    #except:

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
        print(datetime.datetime.now())
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code
    p = [p for p in mp.find()]
    IS_COMPLEATED = True
    dummy_coll = None
    for i in p:
        print(i,'ttttttttttttttttttttttttttttttttttt')
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
