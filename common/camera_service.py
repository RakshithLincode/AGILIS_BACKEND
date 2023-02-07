import cv2
import redis
import json
from utils import RedisKeyBuilderServer,CacheHelper,MongoHelper
import camera_module
import sys
sys.path.insert(1, 'D:/SE_PROJECT/livis-be/livis/livis/')
from settings import REDIS_CLIENT_HOST
from settings import REDIS_CLIENT_PORT
from settings import WORKSTATION_COLLECTION
import threading
import datetime
from setting_keys import *
from setting_keys import original_frame_keyholder
import numpy as  np
import time
import imutils
import base64

def config_file1():
    baumer_ip = []
    baumer_ip.append("192.168.1.4")
    baumer_ip.append("192.168.1.3")
    return baumer_ip

def assign_key():
    mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
    p = [p for p in mp.find()]
    workstation_id = p[0]['_id']
    print(workstation_id)
    data = RedisKeyBuilderServer(workstation_id).workstation_info
    key_list = []
    for cam in data['camera_config']['cameras']:
        camera_index = cam['camera_id']
        camera_name = cam['camera_name']
        key = RedisKeyBuilderServer(workstation_id).get_key(cam['camera_id'],original_frame_keyholder) #WS-01_0_original-frame
        key_list.append(key)
    return key_list

rch = CacheHelper()
baumer_ip = config_file1()
# cam_1 = camera_module.camera('baumer',baumer_ip[0])
# cam_2 = camera_module.camera('baumer',baumer_ip[1])
cam_1 = camera_module.Lucid('222500178','left')
cam_2 = camera_module.Lucid('222500178','right')

while True:
    try:
        s = time.time()
        camera_1 = cam_1.fetch_cameras()
        # camera_1 = cv2.imread('img28.jpg')
        if camera_1 is None:
            print("Baumer_camera_1 is not connected to device")
        # frame_1 = imutils.rotate(frame_1,180)
        camera_2 = cam_2.fetch_cameras()
        # camera_2 = cv2.imread('img32.jpg')
        if camera_2 is None:
            print("Baumer_camera_2 is not connected to device")    
        # frame_2 = imutils.rotate(frame_2,180)
        camera_1 = cv2.resize(camera_1,(500,300))
        camera_2 = cv2.resize(camera_2,(500,300))
        cv2.imshow('frame_1',camera_1)
        cv2.imshow('frame_2',camera_2)
        d = time.time()
        key = assign_key()
        print("elapsed to read: "+ str(d-s))
        rch.set_json({key[0]:camera_1})
        rch.set_json({key[1]:camera_2})
        d = time.time()
        print("elapsed to write"+ str(d-s))
    except Exception as e:
        print(e)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
cv2.destroyAllWindows() 



