import cv2
import redis
import json
from utils import RedisKeyBuilderServer,CacheHelper,MongoHelper

import sys
sys.path.insert(0,"../livis/")
from settings import REDIS_CLIENT_HOST
from settings import REDIS_CLIENT_PORT
from settings import WORKSTATION_COLLECTION
import threading
import datetime
from setting_keys import *
import numpy as  np
import time
import imutils
import base64


def camera_initialize(camera_index):
    cap = cv2.VideoCapture(int(camera_index))
    #cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    #cap.set(cv2.CAP_PROP_FOCUS,int(40))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2160)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
    return cap

def returnCamIdx():

    arr = []
    for i in range(10):
        print(i)
        try:
            cap = cv2.VideoCapture(i)
            ret,frame = cap.read()
            cv2.imwrite(str(i)+'.jpg',frame)
            print('Port :', str(i),' is connected!!')
            arr.append(i)
            cap.release()
        except:
            pass
    return arr

def returnAssignedIdx():
    arr = []
    mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
    p = [p for p in mp.find()]
    data = RedisKeyBuilderServer(workstation_id).workstation_info
    for cam in data['camera_config']['cameras']:
        arr.append(int(cam['camera_id']))
    return arr

cam_t0 = datetime.datetime.now() 
mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)

p = [p for p in mp.find()]

workstation_id = p[0]['_id']

data = RedisKeyBuilderServer(workstation_id).workstation_info
#print(data['camera_config']['cameras'])
#data = RedisKeyBuilderWorkstation().workstation_info

#Calling redis module
rch = CacheHelper()

cam_list = []
key_list = []
cap_list = []

for cam in data['camera_config']['cameras']:
    camera_index = cam['camera_id']
    camera_name = cam['camera_name']
    key = RedisKeyBuilderServer(workstation_id).get_key(cam['camera_id'],original_frame_keyholder) #WS-01_0_original-frame
    cap = camera_initialize(camera_index)
    cap_list.append(cap)
    exec(f'{camera_name} = cap')
    cam_list.append(cam['camera_name'])
    key_list.append(key)


print(cam_list)
print(key_list)
print(cap_list)

cam_t1 = datetime.datetime.now() 
print('Time taken for initiallizing camera :{} secs'.format((cam_t1-cam_t0).total_seconds()))



try:
    while True:
        try:

            for cap, key in zip(cap_list, key_list):

                s = time.time()
                ret, frame = cap.read()

                frame = imutils.rotate(frame,180)
                
                if frame is None:
                    print("got None")
                    print(key)
                    a = 1/0
                    

                
                d = time.time()
                print("elapsed to read: "+ str(d-s))

                rch.set_json({key:frame})
                
                
                d = time.time()
                print("elapsed to write"+ str(d-s))

        except Exception as e:
            print(e)
            print(key)
            
            
            if str(e) == "Timeout reading from socket":
                continue
            elif str(e) == "Timeout writing to socket":
                continue
            else:
            	break
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

except Exception as e:
    print(e)
    ## check for available cap indexes []
    ## if its less than 5 write status in workstation (n cam not connected/not working)
    ## compare index with workstation (if match) do nothing else write status (camera index are wrong)

    lst = returnCamIdx()
    print(lst)
    #if len(lst)<5:
    #    print("$$$$$$$$$$$$$$$   only "+ str(len(lst)) + " are connected, please connect all five cameras")
    #    continue
    assigined_indexes = returnAssignedIdx()
    assigined_indexes.sort()
    lst.sort

    if assigined_indexes == lst:
        #everything is ok
        print("everything is ok")
        pass
    else:
        print("$$$$$$$$$$$$$$$$$$   camera indexes are wrong, Please use " + str(lst))
    


  
cv2.destroyAllWindows() 
