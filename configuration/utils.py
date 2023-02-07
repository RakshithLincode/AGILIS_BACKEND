#from common.utils import MongoHelper, run_in_background
from common.utils import MongoHelper
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
from datetime import datetime


def list_specific_jig(jig_type):

    message = None
    status_code = None
    if jig_type is None:
        message = "jig_type not provided"
        status_code = 400
        return message,status_code
    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code
    p = [i for i in mp.find()]
   
    new_list = []
   
    for i in p:
       
        if i['jig_type'] == jig_type and not i['is_deleted'] == True:
            new_list.append(i)
    return new_list,200


def fetch_individual_component_list_util(data):
    message = None
    status_code = None

    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except Exception as e:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    jig = [p for p in mp.find({"$and" : [{"is_deleted": False}, { "is_deleted" : {"$exists" : True}}]}).sort( "$natural", -1 )]


    kanban_list = []
    for row in jig:
        kanban_list.append(row['kanban'])


    #time intensive statements, must optimise later
    component_list = []
    for comp in kanban_list:
        for c in comp:
            array = c['part_number']
            for j in array:
                if j not in component_list:
                    component_list.append(j)
        

    message = component_list
    status_code = 200
    return message,status_code


def fetch_jig_list_util(data):

    message = None
    status_code = None

    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except Exception as e:
        print(e)
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    jig = [p for p in mp.find({"$and" : [{"is_deleted": False}, { "is_deleted" : {"$exists" : True}}]}).sort( "$natural", -1 )]
    #p = [i for i in mp.find()]

    message = jig
    status_code = 200
    return message,status_code

def add_jig_util(data):

    message = None
    status_code = None
    now = datetime.now()
    current_date_time = now.strftime("%d/%m/%Y %H:%M")

    user_deatils = data['user_info']
    user_name = user_deatils['username']
    user_role = user_deatils['role_name']
    jig_type =  data['jig_type']
    if jig_type is None:
        message = "jig type not provided"
        status_code = 400
        return message,status_code
    
    oem_number = data['oem_number']
    if oem_number is None:
        message = "oem number not provided"
        status_code = 400
        return message,status_code


    heatsink_match = data['heatsink']
    vendor_match = data['vendor_match']
    palette_match = data['palette']
    # if vendor_match is None:
    #    message = "oem number not provided"
    #    status_code = 400
    #    return message,status_code

    kanban = data['kanban']
    if kanban is None:
        message = "kanban not provided"
        status_code = 400
        return message,status_code


    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code


    p = [p for p in mp.find({"$and" : [{"is_deleted": False}, { "is_deleted" : {"$exists" : True}}]}).sort( "$natural", -1 )]


    for i in p:
        if str(i['oem_number']) == str(oem_number):
            message = "oem number already exists"
            status_code = 500
            return message,status_code
    
    full_img = [{
		"cam_name": "extreme_left_camera",
		"regions": []
	},
	{
		"cam_name": "left_camera",
		"regions": []
	},
	{
		"cam_name": "middle_camera",
		"regions": []
	},
	{
		"cam_name": "right_camera",
		"regions": []
	},
	{
		"cam_name": "extreme_right_camera",
		"regions": []
	}
	]
	
    try: 
        collection_obj = {
            'jig_type' : jig_type,
            'oem_number' : oem_number,
            'heatsink_match' :heatsink_match,
            'palette_match' : palette_match,
            'kanban' : kanban,
            'vendor_match':vendor_match,
            'is_deleted' : False,
            'full_img' : full_img,
            'created_on': current_date_time,
            'modified_on': current_date_time,
            'user_name':user_name,
            'user_role':user_role,
        }
        mp.insert(collection_obj)
    except:
        message = "Error adding object to db"
        status_code = 500
        return message,status_code


    message = "success"
    status_code = 200
    return message,status_code

def fetch_specific_jig_util(jig_id):

    message = None
    status_code = None

    if jig_id is None:
        message = "JIG id not provided"
        status_code = 400
        return message,status_code

    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    try:
        dataset = mp.find_one({'_id' : ObjectId(jig_id)})
        if dataset is None:
            message = "jig_id not found in jig collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid jig id"
        status_code = 400
        return message,status_code
    
    #p = [i for i in mp.find()]

    message = dataset
    status_code = 200

    return message,status_code

def update_jig_util(data):

    message = None
    status_code = None
    now = datetime.now()
    current_date_time = now.strftime("%d/%m/%Y %H:%M")

    jig_id = data['jig_id']
    user_deatils = data['user_info']
    user_name = user_deatils['username']
    user_role = user_deatils['role_name']
    if jig_id is None:
        message = "JIG id not provided"
        status_code = 400
        return message,status_code

    jig_type =  data['jig_type']
    if jig_type is None:
        message = "jig type not provided"
        status_code = 400
        return message,status_code
    
    heatsink_match = data['heatsink']
    vendor_match = data['vendor_match']
    palette_match = data['palette']
    #if jig_type is None:
    #    message = "jig type not provided"
    #    status_code = 400
    #    return message,status_code

    oem_number = data['oem_number']
    if oem_number is None:
        message = "oem number not provided"
        status_code = 400
        return message,status_code


    kanban = data['kanban']
    if kanban is None:
        message = "kanban not provided"
        status_code = 400
        return message,status_code
    try:
        is_deleted = data['is_deleted']
    except:
        is_deleted = None
        pass
    #if is_deleted is None:
    #    message = "is_deleted not provided"
    #    status_code = 400
    #    return message,status_code

    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code


    try:
        dataset = mp.find_one({'_id' : ObjectId(jig_id)})
        if dataset is None:
            message = "jig_id not found in jig collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid jig id"
        status_code = 400
        return message,status_code

    if is_deleted is None:
        is_deleted = dataset['is_deleted']

    try: 
        collection_obj = {
            'jig_type' : jig_type,
            'oem_number' : oem_number,
            'heatsink_match' :heatsink_match,
            'palette_match' : palette_match,
            'kanban' : kanban,
            'vendor_match':vendor_match,
            'is_deleted' : False,
            'modified_on': current_date_time,
            'user_name':user_name,
            'user_role':user_role,
            # 'user_id':user_deatils['user_id']

        }
        mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  collection_obj})
    except:
        message = "Error updating object in db"
        status_code = 500
        return message,status_code


    message = data
    status_code = 200

    return message,status_code

def delete_jig_util(data):

    message = None
    status_code = None

    jig_id = data['jig_id']
    if jig_id is None:
        message = "jig id not provided"
        status_code = 400
        return message,status_code

    try:
        mp = MongoHelper().getCollection(JIG_COLLECTION)
    except:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    try:
        dataset = mp.find_one({'_id' : ObjectId(jig_id)})
        if dataset is None:
            message = "jig_id not found in jig collection"
            status_code = 404
            return message,status_code

    except Exception as e:
        message = "Invalid jig id"
        status_code = 400
        return message,status_code

    jig_type=dataset['jig_type']
    oem_number=dataset['oem_number']
    kanban=dataset['kanban']
    is_deleted = True


    try: 
        collection_obj = {
            'jig_type' : jig_type,
            'oem_number' : oem_number,
            'kanban' : kanban,
            'is_deleted' : is_deleted,
        }
        mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  collection_obj})
    except:
        message = "Error deleting object in db"
        status_code = 500
        return message,status_code

    #print(ObjectId(dataset['_id']))
    

    message = dataset
    status_code = 200

    return message,status_code
