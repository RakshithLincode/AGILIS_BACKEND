from common.utils import MongoHelper
from livis.settings import *
from bson import ObjectId
from common.utils import RedisKeyBuilderServer,CacheHelper
from pyzbar import pyzbar

import subprocess
from subprocess import PIPE
from subprocess import call
import cv2

def add_workstation_task(data):
   workstation_name = data.get('workstation_name')
   workstation_ip = data.get('workstation_ip')
   workstation_port = data.get('workstation_port')
   workstation_status = data.get('workstation_status')
   cameras_info = data.get('cameras')
   isdeleted = False
   mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
   collection_obj = {
       'workstation_name' : workstation_name,
       'workstation_ip' : workstation_ip,
       'workstation_port' : workstation_port,
       'workstation_status' : workstation_status,
       'cameras' : cameras_info,
       'isdeleted' : isdeleted
    }
   _id = mp.insert(collection_obj)
   return _id

def delete_workstation_task(wid):
    _id = wid
    mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
    ws = mp.find_one({'_id' : ObjectId(_id)})
    isdeleted = ws.get('isdeleted')
    if not isdeleted:
        ws['isdeleted'] = True
    mp.update({'_id' : ws['_id']}, {'$set' :  ws})
    return _id




# 3 usable api below

def update_workstation_task(data):
    _id = data.get('_id')
    if _id:
        mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
        wc = mp.find_one({'_id' : ObjectId(_id)})
        #workstation_name = data.get('edit_workstation_name')
        #workstation_ip = data.get('edit_workstation_ip')
        #workstation_port = data.get('edit_workstation_port')
        #workstation_status = data.get('edit_workstation_status')
        cameras = data.get('camerasEdit')
        #if workstation_name:
        #    wc['workstation_name'] = workstation_name
        #if workstation_ip:
        #    wc['workstation_ip'] = workstation_ip
        #if workstation_port:
        #    wc['workstation_port'] = workstation_port
        #if workstation_status:
        #    wc['workstation_status'] = workstation_status
        if cameras:
            i=0
            for camera in cameras:
                camera['camera_id'] = camera.pop('edit_camera_id')
                camera['camera_name'] = camera.pop('edit_camera_name')
                i+=1
            wc['camera_config']['cameras'] = cameras
        mp.update({'_id' : wc['_id']}, {'$set' :  wc})
    return _id

def get_workstations_task():
    mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
    workstations = [p for p in mp.find()]
    if workstations:
        return workstations
    else:
        return []

def calibarate_cameras_util():
    #fetch 5 redis cam feed
    #read barcode on all frames
    #assign which cam belongs to which index
    #send back json for ui populate
    
    #print("insode cam calibrate")

    rch = CacheHelper()
    try:
        mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
    except:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    p = [p for p in mp.find()]

    workstation_det = p[0]['camera_config']['cameras']
    workstation_name = p[0]['workstation_name']

    #print(workstation_det)
    #print(workstation_name)
    cam_name = []
    cam_id = []

    for indexes in workstation_det:
        cam_id.append(indexes['camera_id'])
        cam_name.append(indexes['camera_name'])

    redis_key_list = []
    assign = []
    
    #print(cam_id)
    name = 0
    tries = 0
    dummy_dct = {}
    
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_1000"])
    arucoParams = cv2.aruco.DetectorParameters_create()
    
    
    #return "erro",400
    
    while True:
        
        
        for actual_idx in cam_id:
            
            red_key = str(workstation_name)+'_'+str(actual_idx)+'_'+'original-frame'
            #redis_key_list.append(str(workstation_name)+'_'+str(actual_idx)+'_'+'original-frame')

            frame  = rch.get_json(red_key)
            cv2.imwrite("/home/schneider/deployment25Nov/livis/workstations/img/"+str(name)+".jpg",frame)

            #frame1 = base64.b64decode(frame1)
            #frame1 = np.frombuffer(frame1, dtype=np.uint8)
            #frame = cv2.imdecode(frame1, flags=1)

            #barcode read 

            #barcodes = pyzbar.decode(frame)
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,arucoDict,parameters=arucoParams)
            

            #print(barcodes)
            decoded_string = None
        
            #if len(barcodes) > 0:
        
            #    for barcode in barcodes:
            #        decoded_string = str(barcode.data.decode("utf-8"))
        
            #print(decoded_string)
            if ids is None:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,arucoDict,parameters=arucoParams)
                if ids is None:
                    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    ret,frame1 = cv2.threshold(frame1,120,255,cv2.THRESH_OTSU)
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame1,arucoDict,parameters=arucoParams)
                
            if ids is None:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,arucoDict,parameters=arucoParams)
                if ids is None:
                    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    ret,frame1 = cv2.threshold(frame1,120,255,cv2.THRESH_OTSU)
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame1,arucoDict,parameters=arucoParams)
                
            if ids is None:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,arucoDict,parameters=arucoParams)
                if ids is None:
                    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    ret,frame1 = cv2.threshold(frame1,120,255,cv2.THRESH_OTSU)
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame1,arucoDict,parameters=arucoParams)
                            
            if ids is not None:
            
                print("ids is")
                print(ids[0][0])
                #for i in assign:
                #    if i[0] == actual_idx:
                #        assign.append([actual_idx,decoded_string])
                if ids[0][0] == 1:
                    decoded_string = "ONE"
                    dummy_dct[actual_idx] = decoded_string
                elif ids[0][0] == 2:
                    decoded_string = "TWO"
                    dummy_dct[actual_idx] = decoded_string
                elif ids[0][0] == 3:
                    decoded_string = "THREE"
                    dummy_dct[actual_idx] = decoded_string
                elif ids[0][0] == 4:
                    decoded_string = "FOUR"
                    dummy_dct[actual_idx] = decoded_string
                elif ids[0][0] == 5:
                    decoded_string = "FIVE"
                    dummy_dct[actual_idx] = decoded_string
                #print(dummy_dct)
            
        if tries == 105 or len(dummy_dct) == 5:
            print(tries)
            print(dummy_dct)
            break
        else:
            tries = tries + 1
            continue
        
        
        
        
    print(dummy_dct)
    if len(dummy_dct) != 5:
        return "some barcode values are not decoded please try again",400
    
    dummy_dict = {
        "camera_config": {
            "cameras": [{
                "camera_name": "extreme_left_camera",
                "camera_id": "0"
            }, {
                "camera_name": "left_camera",
                "camera_id": "2"
            }, {
                "camera_name": "middle_camera",
                "camera_id": "4"
            }, {
                "camera_name": "right_camera",
                "camera_id": "6"
            }, {
                "camera_name": "extreme_right_camera",
                "camera_id": "8"
            }]
        }
    }

    for key,result in dummy_dct.items():
        if result == "ONE":
            dummy_dict['camera_config']['cameras'][0] = {"camera_name": "extreme_left_camera", "camera_id": str(key)}
        elif result == "TWO":
            dummy_dict['camera_config']['cameras'][1] = {"camera_name": "left_camera", "camera_id": str(key)}
        elif result == "THREE":
            dummy_dict['camera_config']['cameras'][2] = {"camera_name": "middle_camera", "camera_id": str(key)}
        elif result == "FOUR":
            dummy_dict['camera_config']['cameras'][3] = {"camera_name": "right_camera", "camera_id": str(key)}
        elif result == "FIVE":
            dummy_dict['camera_config']['cameras'][4] = {"camera_name": "extreme_right_camera", "camera_id": str(key)}
        else:
            pass
            
    #TODO update mongo db
    
    mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
    p = mp.find_one({'_id' : ObjectId("5f4cce083f8fa13b3271656b")})
    
    mp.update({'_id' : ObjectId("5f4cce083f8fa13b3271656b")}, {'$set' :  dummy_dict})
    import pymongo
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["LIVIS"]
    mycol = mydb["workstations"]
    #mycol.update({"_id":ObjectId("60070cc41ad3c5ef42aeca72")}, {"$unset": {"camera_config":1}},False,True);
    mycol.update({"_id":ObjectId("60070cc41ad3c5ef42aeca72")},{"$set": {"camera_config":{"cameras": dummy_dict['camera_config']['cameras'] } } } )
    

    #TODO restart supervisor camera_service
    #subprocess.Popen(["supervisorctl", "restart", "camera_service"],shell=True)
    cmd0 = "supervisorctl restart camera_service"
    subprocess.Popen(cmd0,shell=True).wait()
    import time
    #time.sleep(50)

    message = dummy_dict
    status_code = 200
    return message,status_code






def get_workstation_config_task(workstation_id):
    _id = ObjectId(workstation_id)
    mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
    p = mp.find_one({'_id' : _id})
    if p:
        return p
    else:
        return {}

