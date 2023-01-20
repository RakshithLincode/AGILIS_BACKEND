from sklearn.metrics import classification_report
from common.utils import MongoHelper, ExperimentDataset
from bson import ObjectId
from livis.celeryy import app
from pathlib import Path
import numpy as np
#from django.conf import settings
from livis import settings 
import os
import pandas as pd
from celery import shared_task
from bson import ObjectId
from livis.settings import *
from livis.constants import *
import shutil


import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


import cv2
import os
import sys
import numpy as np
import random
import shutil
import glob
import xml.etree.ElementTree as ET
import skimage
import uuid

import subprocess
import skimage.io

from subprocess import PIPE
from subprocess import call

class Augment():

    def change_xml(self,path):
        for filename1 in glob.glob( os.path.join(path, '*.xml')):
            tree = ET.parse(filename1)
            tree.find('.//filename').text = filename1.split("/")[-1].replace(".xml",".png")
            tree.write(filename1)


    def change_brightness_and_contrast(self,image):
        num = random.randint(1, 75)
        brightness = num
        contrast = num
        image = np.int16(image)
        image = image * (contrast/127+1) - contrast + brightness
        image = np.clip(image, 0, 255)
        image = np.uint8(image)
        return image,num

    def add_blur(self,image):
        num = random.randrange(1,5,2)
        blurred = cv2.GaussianBlur(image,(num,num),0)
        return blurred,num

    def add_noise(self,image,mode):
        num = random.randrange(1,10,2)
        gimg = skimage.util.random_noise(image, mode=mode)
        return gimg,num

#TODO : handle nones
#TODO : add logging
#TODO : include more stages as mixup, augmentations
def add_experiment(config):
    part_id = config.get('part_id')
    label_list = config.get('label_list')
    experiment_name = config.get('experiment_name', None)
    experiment_type = config.get('experiment_type', None)
    mp = MongoHelper().getCollection(part_id + '_experiment')
    collection_obj = {
            'status' : 'started',
            'message': '',
            'label_list' : label_list,
            'experiment_name' : experiment_name,
            'experiment_type' : experiment_type
    }
    experiment_id = mp.insert(collection_obj)
    return experiment_id,part_id




def get_experiment_status(part_id,experiment_id):
    try:
        mp = MongoHelper().getCollection(part_id + 'experiment')
        return mp.find_one({'_id' : ObjectId(experiment_id)})
        #return {"status" : mp.find_one({'_id' : _id})['status']}
    except:        
        return {}

#@app.task(bind=True)
@shared_task
def process_job_request(config, experiment_id,part_id):
    #print(config)
    #if config['experiment_type'] == 'classification':
    #    return train_fastai(config,experiment_id)
    if config['experiment_type'] == 'ocr':
        return run_monk_train_utils(config, experiment_id,part_id)
        # return None




def get_running_experiment_status(part_id):
    try:
        mp = MongoHelper().getCollection(str(part_id) + '_experiment')
        exp = [i for i in mp.find()]
        list_of_running = []
        for i in exp:
            if i["status"] == "running":
                list_of_running.append(i)
        data={}
        data["running_experiments"] = list_of_running
        return data
        #return {"status" : mp.find_one({'_id' : _id})['status']}
    except:        
        return {}

def get_all_running_experiments_status():
    mp = MongoHelper().getCollection(JIG_COLLECTION)
    parts = [p for p in mp.find({"$and" : [{"is_deleted": False}, { "is_deleted" : {"$exists" : True}}]}).sort( "$natural", -1 )]
    #parts = [p for p in mp.find({"$and" : [{"isdeleted": False}, { "isdeleted" : {"$exists" : True}}]}).skip(skip).limit(limit)]
    list_of_running = []
    for i in parts:
        part_obj_id = i["_id"]
        mp = MongoHelper().getCollection(str(part_obj_id) + '_experiment')
        exp = [i for i in mp.find()]
        for i in exp:
            if i["status"] in ['started',"running"]:
                list_of_running.append(i)
    data={}
    data["running_experiments"] = list_of_running
    print(data,'datttttttttttttttttttttttttttttttttttttttttttttttt')
    return data


def deploy_experiment_util(data):
    part_id = data["part_id"]
    experiment_id = data["experiment_id"]
    workstation_ids = data["workstation_ids"]
    try:
        mp = MongoHelper().getCollection(str(part_id) + '_experiment')
        exp = [i for i in mp.find()]
        for i in exp:
            if str(ObjectId(i['_id']))==str(experiment_id):
                collection_obj = {
                    'deployed':True,
                    'deployed_on_workstations': workstation_ids
                }
                mp.update({'_id' : ObjectId(i['_id'])}, {'$set' : collection_obj})
                return i       
    except Exception as e:   
        print(e)     
        return {}



def get_deployment_list_util():
    return_list = []
    part_collection = MongoHelper().getCollection(settings.PARTS_COLLECTION)
    workstation_collection = MongoHelper().getCollection(settings.WORKSTATION_COLLECTION)
    parts = [p for p in part_collection.find({"$and" : [{"isdeleted": False}, { "isdeleted" : {"$exists" : True}}]}).sort( "$natural", -1 )]
    for part in parts:
        part_id = part['_id']
        mp = MongoHelper().getCollection(str(part_id) + '_experiment')
        exp = [i for i in mp.find()]
        for experiment in exp:
            print(experiment)
            try:
                if 'deployed' in experiment and experiment['deployed']:
                    for dw in experiment['deployed_on_workstations']:
                        ws_name = workstation_collection.find_one({'_id' : ObjectId(dw)})['workstation_name']
                        resp = {
                            "experiment_name" : experiment['experiment_name'],
                            "part_number" : part['part_number'],
                            "experiment_type" : experiment['experiment_type'],
                            "workstation" : ws_name
                        }
                        return_list.append(resp)
            except:
                pass
    return return_list







def generate_folders(new_parts,jig_type,experiment_id):


    # master config in - /home/schneider/Documents/critical_data/master_files/frcnn.config
    # master gen csv script - /home/schneider/Documents/critical_data/master_files/generate_csv_records.py
    # master gen tf records - /home/schneider/Documents/critical_data/master_files/generate_tfrecords.py

    # in /home/schneider/Documents/critical_data/training_exp/{exp_id}/images -- train test and csv
    # in /home/schneider/Documents/critical_data/training_exp/{exp_id}/training -- modified frcnn config file and new labelmap 
    # in /home/schneider/Documents/critical_data/training_exp/{exp_id}/inference_graph
    # in /home/schneider/Documents/critical_data/training_exp/{exp_id}/test.tf_records
    # in /home/schneider/Documents/critical_data/training_exp/{exp_id}/train.tf_records
    # in /home/schneider/Documents/critical_data/training_exp/{exp_id}/tmp 

    images_folder_pth = "/home/schneider/Documents/critical_data/training_exp/" + str(experiment_id) + "/images"
    training_folder_pth = "/home/schneider/Documents/critical_data/training_exp/" + str(experiment_id) + "/training"
    inference_graph_folder_pth = "/home/schneider/Documents/critical_data/training_exp/" + str(experiment_id) + "/inference_graph"
    base_folder_pth = "/home/schneider/Documents/critical_data/training_exp/" + str(experiment_id) #base folder path is also for tf records
    tmp_folder_pth = "/home/schneider/Documents/critical_data/training_exp/" + str(experiment_id) + "/tmp" #delete after test train split

    if not os.path.exists(base_folder_pth):
        os.makedirs(base_folder_pth)
    else:
        shutil.rmtree(base_folder_pth, ignore_errors=True)
        os.makedirs(base_folder_pth)
        
    if not os.path.exists(images_folder_pth):
        os.makedirs(images_folder_pth)
    else:
        shutil.rmtree(images_folder_pth, ignore_errors=True)
        os.makedirs(images_folder_pth)


    if not os.path.exists(training_folder_pth):
        os.makedirs(training_folder_pth)
    else:
        shutil.rmtree(training_folder_pth, ignore_errors=True)
        os.makedirs(training_folder_pth)



    if not os.path.exists(inference_graph_folder_pth):
        os.makedirs(inference_graph_folder_pth)
    else:
        shutil.rmtree(inference_graph_folder_pth, ignore_errors=True)
        os.makedirs(inference_graph_folder_pth)



    if not os.path.exists(tmp_folder_pth):
        os.makedirs(tmp_folder_pth)
    else:
        shutil.rmtree(tmp_folder_pth, ignore_errors=True)
        os.makedirs(tmp_folder_pth)



    crop_train_dir = os.path.join(tmp_folder_pth,'crop_train_dir')
    if not os.path.exists(crop_train_dir):
        os.makedirs(crop_train_dir)
    else:
        shutil.rmtree(crop_train_dir, ignore_errors=True)

    crop_test_dir = os.path.join(tmp_folder_pth,'crop_test_dir')
    if not os.path.exists(crop_test_dir):
        os.makedirs(crop_test_dir) 
    else:
        shutil.rmtree(crop_test_dir, ignore_errors=True)


    annotated_dir = os.path.join(tmp_folder_pth,'annotated')
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir)
    else:
        shutil.rmtree(annotated_dir, ignore_errors=True)

    all_img_dir = os.path.join(tmp_folder_pth,'all_img')
    if not os.path.exists(all_img_dir):
        os.makedirs(all_img_dir)
    else:
        shutil.rmtree(all_img_dir, ignore_errors=True)

    all_xml_dir = os.path.join(tmp_folder_pth,'all_xml')
    if not os.path.exists(all_xml_dir):
        os.makedirs(all_xml_dir)
    else:
        shutil.rmtree(all_xml_dir, ignore_errors=True)

    for i in new_parts:

        class_dir = os.path.join(tmp_folder_pth,str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        else:
            shutil.rmtree(class_dir, ignore_errors=True)

    weights_folder = "/home/schneider/Documents/critical_data/trained_models/all/"+str(experiment_id)+"/saved_model/1/"
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
    else:
        shutil.rmtree(weights_folder, ignore_errors=True)

    


    return weights_folder,images_folder_pth,training_folder_pth,inference_graph_folder_pth,base_folder_pth,tmp_folder_pth,crop_train_dir,crop_test_dir,annotated_dir,all_img_dir,all_xml_dir


def generate_xml(p,all_xml_dir,all_img_dir,new_parts,tmp_folder_pth,jig_id):

    class_label_list = []
    
    for i in p:

        state = i['state']
        regions = i['regions']
        file_path = i['file_path']

        head_tail = os.path.split(file_path)

        xml_file_path = all_xml_dir
        name_of_img = head_tail[1]
        further_split = os.path.split(head_tail[0])
        folder_name = further_split[0]
        name_of_xml = str(name_of_img.split('.')[0]) +".xml"
        

        try:

            if len(regions) != 0:  
                if regions != []:
                    if regions != [[]]:
                        if state == 'updated' or state == 'tagged':
                            #save label in a list 
                            for j in regions:
                                print(str(j['cls']))
                                #print(new_parts)
                                if str(j['cls']) in new_parts:
                                    
                                    im = cv2.imread(file_path)
                                    height, width, depth  = im.shape

                                    class_dir = os.path.join(tmp_folder_pth,str(j['cls']))
                                    #if not os.path.exists(class_dir):
                                    #    os.makedirs(class_dir)
                                    #else:
                                    #    shutil.rmtree(class_dir, ignore_errors=True)

                                    class_label_list.append(str(j['cls']))

                                    #copy file to annotation_dir
                                    shutil.copy2(file_path, class_dir)
                                    full_path_of_xml = os.path.join(class_dir,name_of_xml)
                                    print(full_path_of_xml)
                                    #generate xml directly to annotaion_dir
                                    annotation = ET.Element('annotation')
                                    ET.SubElement(annotation, 'folder').text = str(folder_name)
                                    ET.SubElement(annotation, 'filename').text = str(name_of_img)
                                    ET.SubElement(annotation, 'path').text = str(file_path)


                                    size = ET.SubElement(annotation, 'size')
                                    ET.SubElement(size, 'width').text = str(width)
                                    ET.SubElement(size, 'height').text = str(height)
                                    ET.SubElement(size, 'depth').text = str(depth)

                                    ET.SubElement(annotation, 'segmented').text = '0'
                                    
                                    
                                    x = j["x"]
                                    y = j["y"]
                                    w = j["w"]
                                    h = j["h"]

                                    x0 = x * width
                                    y0 = y * height
                                    x1 = ((x+w) * width)
                                    y1 = ((y+h) * height)
                                    
                                    


                                    label = str(j["cls"])
                                    print("label inside xml is:")
                                    print(label)
                                    if label in new_parts:

                       
                                        ob = ET.SubElement(annotation, 'object')
                                        ET.SubElement(ob, 'name').text = str(label)
                                        ET.SubElement(ob, 'pose').text = 'Unspecified'
                                        ET.SubElement(ob, 'truncated').text = '1'
                                        ET.SubElement(ob, 'difficult').text = '0'

                                        bbox = ET.SubElement(ob, 'bndbox')
                                        ET.SubElement(bbox, 'xmin').text = str(int(x0))
                                        ET.SubElement(bbox, 'ymin').text = str(int(y0))
                                        ET.SubElement(bbox, 'xmax').text = str(int(x1))
                                        ET.SubElement(bbox, 'ymax').text = str(int(y1))
                                
                                        tree = ET.ElementTree(annotation) 
                                        #write xml in annotation folder
                                        with open(full_path_of_xml, "wb") as files : 
        
                                            tree.write(files)
        except Exception as e:
            print(e)
            print("error generating xmls")
            mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
            exp = mp.find_one({'_id' : ObjectId(experiment_id)})
            #exp = exp[0]

            collection_obj = {'status':'stopped',
            'message':'error generating xml'}

            mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})


    return class_label_list


def split_test_train(all_img_dir,all_xml_dir,crop_test_dir,crop_train_dir):

    #copy paste old data to all_img_dir
    #copy paste old data to all_xml_dir
    mp = MongoHelper().getCollection('WEIGHTS')
    p = [i for i in mp.find()]
    p = p[0]

    all_num_classes = p['all_num_classes']
    all_labelmap_pth = p['all_labelmap_pth']
    #"/home/schneider/Documents/critical_data/trained_models/all/new_tf_weights/saved_model"
    all_saved_model_pth = p['all_saved_model_pth']
    all_training_pth = p['all_training_pth']
    old_test_dir = p['old_test_dir']
    old_train_dir = p['old_train_dir']

    #split 80/20 in annotation dir to its respective folders
    from sklearn.model_selection import train_test_split


    image_files = os.listdir(all_img_dir)

    #image_names = [name.replace(".jpg","") for name in image_files]

    train_names,test_names = train_test_split(image_files, test_size=0.2)

    for test_n in test_names:
        xml_name = str(test_n).split(".")[0] + ".xml"

        source = os.path.join(all_img_dir,test_n)
        dest = os.path.join(crop_test_dir,test_n)
        shutil.copy2(source,dest)

        source = os.path.join(all_xml_dir,xml_name)
        dest = os.path.join(crop_test_dir,xml_name)
        shutil.copy2(source,dest)


    for train_n in train_names:
        xml_name = str(train_n).split(".")[0] + ".xml"

        source = os.path.join(all_img_dir,train_n)
        dest = os.path.join(crop_train_dir,train_n)
        shutil.copy2(source,dest)

        source = os.path.join(all_xml_dir,xml_name)
        dest = os.path.join(crop_train_dir,xml_name)
        shutil.copy2(source,dest)

    
    src_files = os.listdir(old_train_dir)
    for file_name in src_files:
        full_file_name = os.path.join(old_train_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, crop_train_dir)

    src_files = os.listdir(old_test_dir)
    for file_name in src_files:
        full_file_name = os.path.join(old_test_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, crop_test_dir)

    #remove all_img_dir (dont need this no more)
    #all_img_dir = os.path.join(temp_dir,'all_img')
    #if not os.path.exists(all_img_dir):
    #    os.makedirs(all_img_dir)
    #else:
    #    shutil.rmtree(all_img_dir, ignore_errors=True)

    #remove all_xml_dir (dont need this no more)
    #all_xml_dir = os.path.join(temp_dir,'all_xml')
    #if not os.path.exists(all_xml_dir):
    #    os.makedirs(all_xml_dir)
    #else:
    #    shutil.rmtree(all_xml_dir, ignore_errors=True)#

def generate_csv(crop_test_dir,crop_train_dir,images_folder_pth):

    def xml_to_csv(path):

        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member[0].text,
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text)
                        )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df


    xml_df_train = xml_to_csv(crop_train_dir)
    xml_df_test = xml_to_csv(crop_test_dir)

    xml_df_train.to_csv((images_folder_pth + "/" + 'train_labels.csv'), index=None)
    xml_df_test.to_csv((images_folder_pth + "/" + 'test_labels.csv'), index=None)


        



def generate_tf_records(new_parts,generate_tf_script_pth,dest,base_folder_pth,crop_test_dir,crop_train_dir,images_folder_pth,experiment_id):

    mp = MongoHelper().getCollection('WEIGHTS')
    p = [i for i in mp.find()]
    p = p[0]

    all_num_classes = p['all_num_classes']
    all_labelmap_pth = p['all_labelmap_pth']
    #"/home/schneider/Documents/critical_data/trained_models/all/new_tf_weights/saved_model"
    all_saved_model_pth = p['all_saved_model_pth']
    all_training_pth = p['all_training_pth']

    old_labelmap_file = all_labelmap_pth

    o = []
    f = open(old_labelmap_file)
    a = f.readlines()

    for i in a:
        if 'name' in i:
            content = str(str(str(i.split(':')[1]).strip()).replace("\'",""))
            o.append(content)
    f.close()

    if new_parts[0] in o:
        merged_list = o
    else:
        merged_list = o + new_parts
    
    #merged_list = list(set(merged_list))

    temp = open(dest, 'w')
    with open(generate_tf_script_pth, 'r') as f:
        for line in f:
            if 'if row_label' in line:
                line = ""
                i = 1
                for k in merged_list:
                    if i ==1:
                        line = line + "    if row_label == \'"+str(k)+"\'"+":"+"\n"+"        return "+str(i)+"\n"
                    else:
                        line = line + "    elif row_label == \'"+str(k)+"\'"+":"+"\n"+"        return "+str(i)+"\n"
                        
                    i=i+1
                line = line + "    else:"+"\n"+"        return 0"+"\n"
            temp.write(line)
            
    #line = "    else:"+"\n"+"        return 0"+"\n"
    #temp.write(line)
    temp.close()

    command1 = "/home/schneider/anaconda3/envs/livis/bin/python "+dest+" --csv_input="+images_folder_pth + "/" + 'train_labels.csv' +" --image_dir="+ crop_train_dir +" --output_path=" + base_folder_pth + "/" + "train.record"
    command2 = "/home/schneider/anaconda3/envs/livis/bin/python "+dest+" --csv_input="+images_folder_pth + "/" + 'test_labels.csv' +" --image_dir="+ crop_test_dir +" --output_path=" + base_folder_pth + "/" + "test.record"
    proc1 = subprocess.Popen(command1,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    proc2 = subprocess.Popen(command2,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    
    
    output1,error1 = proc1.communicate()
    output2,error2 = proc2.communicate()
    
    HAS1 = False
    HAS2 = False
    
    for line in output1.splitlines():
        line = line.decode('ascii')
        print(line)
        if 'Successfully created the TFRecords' in line:
            HAS1 = True
            break
    
    for line in output2.splitlines():
        line = line.decode('ascii')
        print(line)
        if 'Successfully created the TFRecords' in line:
            HAS2 = True
            break 

    if HAS1:
        pass
    else:
        #error in train 
        mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
        exp = mp.find_one({'_id' : ObjectId(experiment_id)})
       

        collection_obj = {'status':'stopped',
        'message':'error in tf records - train :'+str(e)}

        mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})

    
    if HAS2:
        pass
    else:
        #error in test   
        mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
        exp = mp.find_one({'_id' : ObjectId(experiment_id)})
        

        collection_obj = {'status':'stopped',
        'message':'error in tf records - test :'+str(e)}

        mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})





    #os.system("/home/schneider/anaconda3/envs/livis/bin/python "+dest+" --csv_input="+images_folder_pth + "/" + 'train_labels.csv' +" --image_dir="+ crop_train_dir +" --output_path=" + base_folder_pth + "/" + "train.record")
    
    #os.system("/home/schneider/anaconda3/envs/livis/bin/python "+dest+" --csv_input="+images_folder_pth + "/" + 'test_labels.csv' +" --image_dir="+ crop_test_dir +" --output_path=" + base_folder_pth + "/" + "test.record")
 
def generate_labelmap(new_parts,labelmap_path,weights_folder,training_folder_pth,base_folder_pth):

    mp = MongoHelper().getCollection('WEIGHTS')
    p = [i for i in mp.find()]
    p = p[0]
    
    
    labelmap_full_pth = training_folder_pth + "/" + 'labelmap.pbtxt'
    all_num_classes = p['all_num_classes']
    all_labelmap_pth = p['all_labelmap_pth']
    #"/home/schneider/Documents/critical_data/trained_models/all/new_tf_weights/saved_model"
    all_saved_model_pth = p['all_saved_model_pth']
    all_training_pth = p['all_training_pth']


    num_classes = str(int(int(all_num_classes) + int(len(new_parts))))
    head_tail = os.path.split(weights_folder)
    new_weights_pth = head_tail[0]

    new_label_file = labelmap_full_pth
    old_labelmap_file = all_labelmap_pth
    merged_labelmap_file = training_folder_pth + "/"+ 'labelmap.pbtxt'
    shutil.copyfile(old_labelmap_file,merged_labelmap_file)

    o = 0
    f = open(old_labelmap_file)
    a = f.readlines()

    for i in a:
        if 'item' in i:
            o = o + 1
    
    f.close()

    idx = o + 1
    
    
    old_comp_lst = []
    f = open(old_labelmap_file)
    a = f.readlines()

    for i in a:
        if 'name' in i:
            d = str(str(i.split(":")[1]).replace("\'","")).strip()
            old_comp_lst.append(d)
    
    f.close()
    
   
    if new_parts[0] in old_comp_lst:
        pass
    else:
    
        for components in new_parts:

            with open(merged_labelmap_file,'a') as fp:
                fp.write('item {\n  id: '+str(idx)+'\n'+'  name: '+"\'"+components+"\'"+"\n}\n")
            idx = idx + 1

    #labelmap_loc = labelmap_path + "/"+ 'labelmap.pbtxt'


    return labelmap_full_pth

def generate_config(master_config_pth,dest,experiment_id,training_folder_pth,base_folder_pth,crop_test_dir,new_parts,jig_id):

    
    mp = MongoHelper().getCollection('WEIGHTS')
    p = [i for i in mp.find()]
    p = p[0]


    all_num_classes = p['all_num_classes']

    all_training_pth = p['all_training_pth']
    ckpt = None
    last_trained_ckpt = all_training_pth + "/" + "checkpoint"
    f = open(last_trained_ckpt,'r')

    try:  
        a = f.readlines()
        for i in a:
            if 'model_checkpoint_path:' in i:
                c = str(str(i).split(':')[1]).strip()
                c = c.replace('\"',"")
                if '-' in c:
                    ckpt = int(str(c.split('-')[1]).replace('\"',""))
                else:
                    ckpt = 0
        f.close()

    except Exception as e:
        print(e)
        f.close()
        mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
        exp = mp.find_one({'_id' : ObjectId(experiment_id)})
        
        err_msg = 'error in fetching ckpt :'+str(e)

        collection_obj = {'status':'stopped',
        'message':err_msg}

        mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})

    mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
    exp = mp.find_one({'_id' : ObjectId(experiment_id)})

    statuss = exp['status']
    messagee =  exp['message']

    if statuss == 'started':

        #copy paste from prev to this 

        src_files = os.listdir(all_training_pth)
        print(src_files)
        for file_name in src_files:
        
            if file_name == "labelmap.pbtxt":
                continue
            elif file_name == "faster_rcnn_inception_v2_pets.config":
                continue
            elif file_name == "checkpoint":
                continue  
            elif file_name == "graph.pbtxt":
                continue
            elif file_name == "pipeline.config":
                continue
            else:
                full_file_name = os.path.join(all_training_pth, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, training_folder_pth)
 
        if ckpt == 0:
            fine_tune_checkpoint = training_folder_pth + "/" + "model.ckpt"
        else:
            fine_tune_checkpoint = training_folder_pth + "/" + "model.ckpt-" + str(ckpt)

        train_record_pth = base_folder_pth + "/" + "train.record"
        test_record_pth = base_folder_pth + "/" + "test.record"

        labelmap_path = training_folder_pth + "/" + 'labelmap.pbtxt'

        l = os.listdir(crop_test_dir)
        l = len(l)
        num_examples = str(int(l/2))

        num_classes = str(int(int(all_num_classes) + int(len(new_parts))))

        file1=open(master_config_pth,"r")
        file2=open(dest,"w")
    
        for line in file1.readlines():
            if 'num_classes' in line:
                file2.write("    num_classes: "+num_classes+"\n")
            elif 'fine_tune_checkpoint' in line:
                file2.write("  fine_tune_checkpoint: "+ '\"' + fine_tune_checkpoint+ '\"' +"\n")
            elif 'train.record' in line:
                file2.write("    input_path: "+ '\"' + train_record_pth+ '\"' +"\n")
            elif 'test.record' in line:
                file2.write("    input_path: "+ '\"' + test_record_pth+ '\"' +"\n")
            elif 'num_examples' in line:
                file2.write("  num_examples: "+ num_examples +"\n")
            elif 'label_map_path:' in line:
                file2.write("  label_map_path: "+ '\"' + labelmap_path + '\"' +"\n")
            else:
                file2.write(line)

        file1.close()
        file2.close()


def gen_augmentations(all_img_dir,all_xml_dir,new_parts_total,class_dir):

    CONTRAST_BRIGHTNESS = False
    NOISE = True
    BLUR = False
    Total = 164

    folder_path = class_dir

    Aug = Augment()

    list_of_images = os.listdir(folder_path)

    sub_total = None

    #if CONTRAST_BRIGHTNESS is True and NOISE is True and BLUR is True:
    #    sub_total = int(int(Total)/3)
    #elif CONTRAST_BRIGHTNESS is True and NOISE is True or CONTRAST_BRIGHTNESS is True and BLUR is True or NOISE is True and BLUR is True:
    #    sub_total = int(int(Total)/2)
    #else:
    sub_total = int(Total)

    if NOISE:

        start = 1
        path = folder_path
        while(start<=sub_total):
            random_filename = random.choice([x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))])
            #print(random_filename)
            if str(random_filename).split('.')[1] == 'xml':
                continue
            if '_n_' in random_filename:
                continue
            i = random_filename
            if '/' in i:
                i = str(i).split('/')[-1]
            filename = str(i).split('.')[0]
            ext = str(i).split('.')[1]
            if ext == 'jpg' or ext == 'png':
                image,num = Aug.add_noise(skimage.io.imread(folder_path+'/'+i)/255.0,"speckle")
                num = str(uuid.uuid4())
                num = str(str(num).replace("-",""))

                skimage.io.imsave(folder_path+"/"+filename+'_n_'+str(num)+'.'+ext,image)
                original = folder_path+"/"+filename+'.xml'
                target = folder_path+"/"+filename+'_n_'+str(num)+'.xml'
                shutil.copyfile(original, target)
            start+=1

    if BLUR:

        start = 1
        path = folder_path
        while(start<=sub_total):
            random_filename = random.choice([x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))])
            #print(random_filename)
            if str(random_filename).split('.')[1] == 'xml':
                continue
            if '_blur_' in random_filename:
                continue
            i = random_filename
            if '/' in i:
                i = str(i).split('/')[-1]
            filename = str(i).split('.')[0]
            ext = str(i).split('.')[1]
            if ext == 'jpg' or ext == 'png':
                image,num = Aug.add_blur(cv2.imread(folder_path+'/'+i))
                num = str(uuid.uuid4())
                cv2.imwrite(folder_path+"/"+filename+'_blur_'+str(num)+'.'+ext,image)
                original = folder_path+"/"+filename+'.xml'
                target = folder_path+"/"+filename+'_blur_'+str(num)+'.xml'
                shutil.copyfile(original, target)
            start+=1
        

    Aug.change_xml(folder_path)









def run_monk_train_utils(data,experiment_id,part_id):
    message = None
    status_code = None


    jig_id = data['part_id']
    #lr = data['lr']
    #batch_size = data['batch_size']
    #num_steps = data['num_steps']
    model_type = data['model_type']
    new_parts = data['new_parts']
    print("new_parts are")
    print(new_parts)
    


    if new_parts is None:
        print("new part data not provided")
        sys.exit()
    
    if len(new_parts) == 0:
        print("no new parts selected")
        sys.exit()
    new_p = []
        
    for k in new_parts:
        new_p.append(str(k.strip()))
    
    new_parts = new_p.copy()
    

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
    jig_type = dataset['jig_type']
    _id = str(dataset['_id'])


    mp = MongoHelper().getCollection(str(dataset['_id']))
    p = [i for i in mp.find()]


    if len(p) == 0:
        print("no data on images found in specified collection")
        sys.exit()

    print("generating folders")
    weights_folder,images_folder_pth,training_folder_pth,inference_graph_folder_pth,base_folder_pth,tmp_folder_pth,crop_train_dir,crop_test_dir,annotated_dir,all_img_dir,all_xml_dir = generate_folders(new_parts,jig_type,experiment_id)
    print("folder generation done !")
    
    
    print("\n")
    print("generating XML")
    class_label_list =  generate_xml(p,all_xml_dir,all_img_dir,new_parts,tmp_folder_pth,jig_id)
    print("XML generated !")
    print("\n")
    
    mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
    exp = mp.find_one({'_id' : ObjectId(experiment_id)})
    #exp = exp[0]
    
    print(exp)

    statuss = exp['status']
    messagee =  exp['message']

    if statuss == 'started':
        

        if len(class_label_list) == 0:
            #no xml generated
            print("no xml generated -- no annotations done")
            mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
            exp = mp.find_one({'_id' : ObjectId(experiment_id)})
            #exp = exp[0]

            collection_obj = {'status':'stopped',
            'message':'no annotations done'}

            mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})
            #make status as stopped and throw error


        mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
        exp = mp.find_one({'_id' : ObjectId(experiment_id)})
        #exp = exp[0]

        statuss = exp['status']
        messagee =  exp['message']

        if statuss == 'started':

            for new_c in new_parts:
                class_dir = os.path.join(tmp_folder_pth,str(new_c))
                l = os.listdir(class_dir)

                if len(l) != 72:
                    print("less than 36 annotations found in selected label")
                    mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
                    exp = mp.find_one({'_id' : ObjectId(experiment_id)})
                    

                    collection_obj = {'status':'stopped',
                    'message':'less than 36 annotations found'}

                    mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})


            mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
            exp = mp.find_one({'_id' : ObjectId(experiment_id)})
            #exp = exp[0]

            statuss = exp['status']
            messagee =  exp['message']

            if statuss == 'started':


                ############################################ stop docker container and check if part is already trained in labelmap of old check  later
                for new_c in new_parts:
                    class_dir = os.path.join(tmp_folder_pth,str(new_c))
                    print("starting augmentations")
                    gen_augmentations(all_img_dir,all_xml_dir,len(new_parts),class_dir)
                    print("done augmentations")
                    

                    l = os.listdir(class_dir)
                    #print(l)
                    for listd in l:
                    
                        if '/' in listd:
                            i = str(listd).split('/')[-1]
                        else:
                            i = listd
                            filename = str(i).split('.')[0]
                            ext = str(i).split('.')[1]
                        
                        
                        if ext == "xml":
                            original = class_dir+"/"+filename+'.xml'
                            target = all_xml_dir+"/"+filename+'.xml'
                            print("in xml")
                            print(original)
                            print(target)
                            shutil.copyfile(original, target)
                        else:
                            original = class_dir+"/"+filename+'.png'
                            target = all_img_dir+"/"+filename+'.png'
                            print("in png")
                            print(original)
                            print(target)
                            shutil.copyfile(original, target)

                print("going into test train split")
                split_test_train(all_img_dir,all_xml_dir,crop_test_dir,crop_train_dir)
                print("ended performing test train split")
                
                try:
                    generate_csv(crop_test_dir,crop_train_dir,images_folder_pth)
                except Exception as e:
                    print("error in generating csv")
                    print(e)
                    mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
                    exp = mp.find_one({'_id' : ObjectId(experiment_id)})

                    collection_obj = {'status':'stopped',
                    'message':'error generating xml :'+str(e)}

                    mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})



                mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
                exp = mp.find_one({'_id' : ObjectId(experiment_id)})

                statuss = exp['status']
                messagee =  exp['message']

                if statuss == 'started':
                    #stop docker containers if running
                    proc = subprocess.run(['docker','ps','-aq'],check=True,stdout=PIPE,encoding='ascii')
                    container_ids = proc.stdout.strip().split()
                    if container_ids:
                        subprocess.run(['docker','stop']+container_ids,check=True)
                        subprocess.run(['docker','rm']+container_ids,check=True)
                    print("killed....")
                    print("killing all flask supervisorssss")
                    cmd0 = "supervisorctl stop flask_service_5000"
                    cmd1 = "supervisorctl stop flask_service_5001"
                    cmd2 = "supervisorctl stop flask_service_5002"
                    cmd3 = "supervisorctl stop flask_service_5003"
                    cmd4 = "supervisorctl stop flask_service_5004"
                    
                    cmd5 = "supervisorctl stop flask_service_5005"
                    cmd6 = "supervisorctl stop flask_service_5006"
                    cmd7 = "supervisorctl stop flask_service_5007"
                    cmd8 = "supervisorctl stop flask_service_5008"

                    
                    subprocess.Popen(cmd0,shell=True).wait()
                    subprocess.Popen(cmd1,shell=True).wait()
                    subprocess.Popen(cmd2,shell=True).wait()
                    subprocess.Popen(cmd3,shell=True).wait()
                    subprocess.Popen(cmd4,shell=True).wait()
                    
                    subprocess.Popen(cmd5,shell=True).wait()
                    subprocess.Popen(cmd6,shell=True).wait()
                    subprocess.Popen(cmd7,shell=True).wait()
                    subprocess.Popen(cmd8,shell=True).wait()
                    print("all flask killed...")
            
                    #gen tf records
                    generate_tf_script_pth = "/home/schneider/Documents/critical_data/master_files/generate_tfrecord.py"
                    dest = tmp_folder_pth + "/" + "generate_tfrecord.py"

                    #shutil.copyfile(generate_tf_script_pth,dest)
                    print("entering gen tf records")
                    generate_tf_records(new_parts,generate_tf_script_pth,dest,base_folder_pth,crop_test_dir,crop_train_dir,images_folder_pth,experiment_id)
                    print("successfully gen tf records")
                    
                    
                    mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
                    exp = mp.find_one({'_id' : ObjectId(experiment_id)})
                    

                    statuss = exp['status']
                    messagee =  exp['message']

                    if statuss == 'started':

                        #generate labelmap in training folder 
                        print("generating labelmap")
                        
                        labelmap_full_pth = generate_labelmap(new_parts,training_folder_pth,weights_folder,training_folder_pth,base_folder_pth)
                        print("finished gen labelmap")
                        
                        #frcnn config change 
                        master_config_pth = "/home/schneider/Documents/critical_data/master_files/faster_rcnn_inception_v2_pets.config"
                        dest = training_folder_pth + "/" + "faster_rcnn_inception_v2_pets.config"
                        print("generating config file")
                        generate_config(master_config_pth,dest,experiment_id,training_folder_pth,base_folder_pth,crop_test_dir,new_parts,jig_id) 
                        print("finished config file")
                        
                        mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
                        exp = mp.find_one({'_id' : ObjectId(experiment_id)})

                        statuss = exp['status']
                        messagee =  exp['message']

                        if statuss == 'started':
                            #kiill prev tf board
                            comd = "lsof -i tcp:6006 | awk 'NR!=1 {print $2}' | xargs kill"
                            subprocess.Popen(comd,shell=True)
                            #http://localhost:6006/
                            print("launching tensorboard")
                            command0 = '/home/schneider/anaconda3/envs/livis/bin/tensorboard --logdir='+training_folder_pth
                            subprocess.Popen(command0,shell=True)
                            print("tensorboard launched")

                            command1 = '/home/schneider/anaconda3/envs/livis/bin/python /home/schneider/deployment25Nov/livis/models/research/object_detection/train.py --logtostderr --train_dir='+training_folder_pth+' ' +'--pipeline_config_path='+dest
                            
                            print("trining is :::::")
                            print(command1)
                            subprocess.Popen(command1,shell=True).wait()
                            print("training DONE!!!!!!!!!!!!!!!!")
                            
                           

                            ckpt = None
                            last_trained_ckpt = training_folder_pth + "/" + "checkpoint"
                            f = open(last_trained_ckpt,'r')

                            try:  
                                a = f.readlines()
                                for i in a:
                                    if 'model_checkpoint_path:' in i:
                                        c = str(str(i).split(':')[1]).strip()
                                        c = c.replace('\"',"")
                                        ckpt = int(str(c.split('-')[1]).replace('\"',""))
                                f.close()

                            except Exception as e:
                                f.close()
                                mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
                                exp = mp.find_one({'_id' : ObjectId(experiment_id)})

                                collection_obj = {'status':'stopped', 'message':
                                'error in fetching ckpt after training:'+str(e)}

                                mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})
                                
                            if int(ckpt) == 0 or int(ckpt) != 5000:
                                mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
                                exp = mp.find_one({'_id' : ObjectId(experiment_id)})

                                collection_obj = {'status':'stopped', 'message':
                                'error in training or not fully trained:'}
                                mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})
                                

                            mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
                            exp = mp.find_one({'_id' : ObjectId(experiment_id)})

                            statuss = exp['status']
                            messagee =  exp['message']

                            if statuss == 'started':

                                command2 = '/home/schneider/anaconda3/envs/livis/bin/python /home/schneider/deployment25Nov/livis/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path '+ dest +' --trained_checkpoint_prefix '+ training_folder_pth + '/model.ckpt-' + str(ckpt) + ' --output_directory '+inference_graph_folder_pth
                                subprocess.Popen(command2,shell=True).wait()

                                #transfer inf grp ----  saved_model.pb and variables folder to weights folder 
                                inside_saved_models_pth = inference_graph_folder_pth+'/saved_model/'
                                src_files = os.listdir(inside_saved_models_pth)
                                for file_name in src_files:
                                    full_file_name = os.path.join(inside_saved_models_pth, file_name)
                                    if os.path.isfile(full_file_name):
                                        shutil.copy(full_file_name, weights_folder)


                                #if success - update -- all_num_classes , all_label

                                mp = MongoHelper().getCollection('WEIGHTS')
                                p = [i for i in mp.find()]
                                p = p[0]

                                all_num_classes = p['all_num_classes']
                                all_labelmap_pth = p['all_labelmap_pth']
                                #"/home/schneider/Documents/critical_data/trained_models/all/new_tf_weights/saved_model"
                                all_saved_model_pth = p['all_saved_model_pth']
                                all_training_pth = p['all_training_pth']


                                num_classes = str(int(int(all_num_classes) + int(len(new_parts))))
                                head_tail = os.path.split(weights_folder)
                                new_weights_pth = head_tail[0]
                                head_tail = os.path.split(new_weights_pth)
                                new_weights_pth = head_tail[0]
                                
                                
                                """
                                new_label_file = labelmap_full_pth
                                old_labelmap_file = all_labelmap_pth
                                merged_labelmap_file = base_folder_pth + "/"+ 'labelmap.pbtxt'
                                shutil.copyfile(old_labelmap_file,merged_labelmap_file)

                                o = 0
                                f = open(old_labelmap_file)
                                a = f.readlines()

                                for i in a:
                                    if 'item' in i:
                                        o = o + 1
                                
                                f.close()

                                idx = o

                                for components in new_parts:

                                    with open(merged_labelmap_file,'a') as fp:
                                        fp.write('item {\n  id: '+str(idx)+'\n'+'  name: '+"\'"+components+"\'"+"\n}\n")
                                    idx = idx + 1

                                """
                                labelmap_full_pth = training_folder_pth + "/" + 'labelmap.pbtxt'
                                collection_obj = {
                                    'all_num_classes' : num_classes,
                                    'all_labelmap_pth' : labelmap_full_pth,
                                    'all_saved_model_pth' : new_weights_pth,
                                    'all_training_pth' : training_folder_pth,
                                    'old_test_dir': crop_test_dir,
                                    'old_train_dir': crop_train_dir
                                }

                                mp.update({'_id' : ObjectId(p['_id'])}, {'$set' : collection_obj})
                                
                                mp = MongoHelper().getCollection(str(jig_id) + '_experiment')
                                exp = mp.find_one({'_id' : ObjectId(experiment_id)})

                                collection_obj = {'status':'trained', 'message':
                                'train success'}

                                mp.update({'_id' : ObjectId(exp['_id'])}, {'$set' : collection_obj})


def get_trained_list_util():
    mp = MongoHelper().getCollection('WEIGHTS')
    exp = [i for i in mp.find()]
    all_labelmap_pth = exp[0]['all_labelmap_pth']
    with open(all_labelmap_pth)as f:
        a = f.readlines()
        b = []
        c = []
        for i in a:
            if 'name' in i:
                b.append(i)
        for i in b:
            j = i.strip()
            j = str(j).replace("name: ","")
            j = str(j).replace("\'","")
            if 'K75T60' in j:
                j = "K75T60_matte"
            if '40N120FL2' in j:
                j = "40N120FL2_big"
            if j not in c:
                c.append(j)

    return c


