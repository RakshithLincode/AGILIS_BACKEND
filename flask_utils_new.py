from paddleocr import PaddleOCR,draw_ocr 
import os
import matplotlib.pyplot as plt
# %matplotlib inline
import cv2
import torch
import pandas as pd
from pymongo import MongoClient
from livis import settings as s
from iteration_utilities import unique_everseen
from PaddleOCR.tools.infer import predict_system
def singleton(cls):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

@singleton
class MongoHelper:
    client = None
    def __init__(self):
        if not self.client:
            self.client = MongoClient(host=s.MONGO_SERVER_HOST, port=s.MONGO_SERVER_PORT)
        self.db = self.client[s.MONGO_DB]

    def getDatabase(self):
        return self.db

    def getCollection(self, cname, create=False, codec_options=None):
        _DB = s.MONGO_DB
        DB = self.client[_DB]
        if cname in s.MONGO_COLLECTIONS:
            if codec_options:
                return DB.get_collection(s.MONGO_COLLECTIONS[cname], codec_options=codec_options)
            return DB[s.MONGO_COLLECTIONS[cname]]
        else:
            return DB[cname]
            
def model_torch(img_path):
	img = cv2.imread(img_path)
	final_img = img.copy()
	model = torch.hub.load("D:/Segmentatin_yolo/LINCODE_AI_WORKER/segment/LINCODE_AI/",'custom',path="D:/24JAN/best.pt",source = 'local',force_reload=True,autoshape=True)
	model.conf = 0.25
	model.iou = 0.1
	results = model(img,size=1280)
	results.xyxy[0] 
	df = pd.DataFrame(results.pandas().xyxy[0]).sort_values('xmin')
	df_gen = df.loc[df.name == 'IGBT']
	return df_gen,final_img

def get_kanban(oem_number):
    mp = MongoHelper().getCollection("jig")
    for x in mp.find():
        if x['oem_number'] == oem_number:
            kanban = x.get('kanban')
            return kanban

def check_kanban(actual_value,predicted_value):
    position = {'position_present':[],'position_absent':[],'status':[]}
    position_actual = []
    position_predict = []
    isAccepted  = []
    status = []
    for actual,predicted in zip(actual_value,predicted_value):
        if actual['part_number'] == predicted['part_number']:
            position['position_present'].append(actual['position'])
            isAccepted.append('Accepted')  
        else:
            position['position_absent'].append(actual['position'])
            isAccepted.append('Rejected')  
    for value in actual_value:
        position_actual.append(value['position'])
    for value in predicted_value:
        position_predict.append(value['position']) 
    for i in position_actual:
        if i in position_predict:
            print(i)
        else:
            position['position_absent'].append(i)
            isAccepted.append('Rejected')  
    if 'Rejected' in isAccepted:
        position['status'].append('Rejected')
    else:
        position['status'].append('Accepted')
    print(isAccepted)		
    return  position          
    
def get_ocr(img_path,oem_number):
    count = 1
    results_label = []
    df_gen,final_img = model_torch(img_path)
    for index, rows in df_gen.iterrows():
        ymin = int(rows["ymin"]) - 20
        xmax = int(rows["xmax"]) + 40
        ymax = int(rows["ymax"]) + 20 
        xmin = int(rows["xmin"]) - 40
        img_crp_save = final_img[ymin:ymax,xmin:xmax]
        dt_boxes, rec_res, time_dict = predict_system.lincode_ocr(img_crp_save)   
        return dt_boxes, rec_res, time_dict
        # count = count + 1
