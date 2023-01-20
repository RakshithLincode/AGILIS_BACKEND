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
            'label_list' : label_list,
            'experiment_name' : experiment_name,
            'experiment_type' : experiment_type
    }
    experiment_id = mp.insert(collection_obj)
    return experiment_id

  


def get_experiment_status(part_id,experiment_id):
    try:
        mp = MongoHelper().getCollection(part_id + 'experiment')
        return mp.find_one({'_id' : ObjectId(experiment_id)})
        #return {"status" : mp.find_one({'_id' : _id})['status']}
    except:        
        return {}

#@app.task(bind=True)
@shared_task
def process_job_request(config, experiment_id):
    #print(config)
    #if config['experiment_type'] == 'classification':
    #    return train_fastai(config,experiment_id)
    if config['experiment_type'] == 'detection':
        return run_monk_train_utils(config, experiment_id)
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









#schneider training monk ai
"""
def monk_list_models_utils(data):
    

    message = None
    status_code = None
    '''
    import os 
    this_dir = os.getcwd()
    full_pth = this_dir + "/training/"
    os.chdir(full_pth)
    os.system("sudo sh get_model_list.sh > models.txt")
    f = open('models.txt','r')
    l = []
    for i in f:
        if "Model" in i:
            line = str(i).strip()
            line_f = line.split(':')[1]
            l.append(line_f)
    f.close()
    message = l
    status_code = 200
    os.chdir(this_dir)
    '''

    import sys
    sys.path.append("livis/Monk_Object_Detection/13_tf_obj_2/lib/")
    from livis.Monk_Object_Detection.tf_obj_2.lib.train_detector import Detector
    gtf = Detector()

    message = gtf.list_models()

    message = ["ssd_mobilenet_v2_320", "ssd_mobilenet_v1_fpn_640",
                "ssd_mobilenet_v2_fpnlite_320", "ssd_mobilenet_v2_fpnlite_640",
                "ssd_resnet50_v1_fpn_320", "ssd_resnet50_v1_fpn_640",
                "ssd_resnet101_v1_fpn_320", "ssd_resnet101_v1_fpn_640",
                "ssd_resnet152_v1_fpn_320", "ssd_resnet152_v1_fpn_640",
                "faster_rcnn_resnet50_v1_640", "faster_rcnn_resnet50_v1_1024",
                "faster_rcnn_resnet101_v1_640", "faster_rcnn_resnet101_v1_1024",
                "faster_rcnn_resnet152_v1_640", "faster_rcnn_resnet152_v1_1024",
                "faster_rcnn_inception_resnet_v2_640", "faster_rcnn_inception_resnet_v2_1024",
                "efficientdet_d0", "efficientdet_d1", 
                "efficientdet_d2", "efficientdet_d3",
                "efficientdet_d4", "efficientdet_d5",
                "efficientdet_d6", "efficientdet_d7"
                ]

    return message,status_code
"""

def train_exists():

    #check is_trained value in jig_id collection
    #finetune here (use previous trained data and finetune upon it)
    jig_id = data['jig_id']
    


    message = pth
    status_code = 200
    return message,status_code



def run_monk_train_utils(data,experiment_id):
    message = None
    status_code = None

    # open _jigid to get all trainable images 
    # goto that folder 
    # create tmp folder in that - crop_train_img, crop_test_img, crop_train_xml,crop_test_xml,annotated(contains all annotated images),classes(contains classes.txt), tf_records(contains both train and test tf)
    # copy all those have regions to the annotated_dir saving the labels in a list (used for classes.txt)
    # generate xml to entire folder (api already exist in -- export annotations)
    # split 80 20 only for those which have annotation
    # keep in spererate folders - train images, test images, train annotation , test annotation
    # make a class file 
    # run generate tf records 
    # run train command and change path where saved weights are stored in celery 


    jig_id = data['jig_id']
    lr = data['lr']
    batch_size = data['batch_size']
    num_steps = data['num_steps']
    model_type = data['model_type']
    

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
    _id = str(dataset['_id'])


    mp = MongoHelper().getCollection(str(dataset['_id']))
    p = [i for i in mp.find()]


    if len(p) == 0:
        message = "no records found in specified collection"
        status_code = 400
        return message,status_code


    base_path =  os.path.join('/critical_data/')
    if oem_number is None:
        this_model_pth = str(jig_type)
    else:
        this_model_pth = str(jig_type) + str(oem_number)

    dir_path = os.path.join(base_path,this_model_pth)

    dir_crop_path = os.path.join(dir_path,'crops')

    temp_dir = os.path.join(dir_crop_path,'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)

    #clear out previous train test path if exist and create new empty folder

    crop_train_img_dir = os.path.join(temp_dir,'crop_train_img')
    if not os.path.exists(crop_train_img_dir):
        os.makedirs(crop_train_img_dir)
    else:
        shutil.rmtree(crop_train_img_dir, ignore_errors=True)

    crop_test_img_dir = os.path.join(temp_dir,'crop_test_img')
    if not os.path.exists(crop_test_img_dir):
        os.makedirs(crop_test_img_dir) 
    else:
        shutil.rmtree(crop_test_img_dir, ignore_errors=True)

    crop_train_xml_dir = os.path.join(temp_dir,'crop_train_xml')
    if not os.path.exists(crop_train_xml_dir):
        os.makedirs(crop_train_xml_dir)
    else:
        shutil.rmtree(crop_train_xml_dir, ignore_errors=True)

    crop_test_xml_dir = os.path.join(temp_dir,'crop_test_xml')
    if not os.path.exists(crop_test_xml_dir):
        os.makedirs(crop_test_xml_dir)
    else:
        shutil.rmtree(crop_test_xml_dir, ignore_errors=True)

    annotated_dir = os.path.join(temp_dir,'annotated')
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir)
    else:
        shutil.rmtree(annotated_dir, ignore_errors=True)

    all_img_dir = os.path.join(temp_dir,'all_img')
    if not os.path.exists(all_img_dir):
        os.makedirs(all_img_dir)
    else:
        shutil.rmtree(all_img_dir, ignore_errors=True)

    all_xml_dir = os.path.join(temp_dir,'all_xml')
    if not os.path.exists(all_xml_dir):
        os.makedirs(all_xml_dir)
    else:
        shutil.rmtree(all_xml_dir, ignore_errors=True)

    classes_dir = os.path.join(temp_dir,'classes')
    if not os.path.exists(classes_dir):
        os.makedirs(classes_dir)
    else:
        shutil.rmtree(classes_dir, ignore_errors=True)

    tf_records_dir = os.path.join(temp_dir,'tf_records')
    if not os.path.exists(tf_records_dir):
        os.makedirs(tf_records_dir)
    else:
        shutil.rmtree(tf_records_dir, ignore_errors=True)
    
    export_dir = os.path.join(temp_dir,'export_dir')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    else:
        shutil.rmtree(export_dir, ignore_errors=True)

    

    class_label_list = []
    
    for i in p:

        state = i['state']
        regions = i['regions']
        file_path = i['file_path']

        head_tail = os.path.split(img_file_path)

        xml_file_path = annotated_dir
        name_of_img = head_tail[1]
        name_of_xml = str(name_of_img.split('.')[0]) +".xml"
        full_path_of_xml = os.path.join(xml_file_path,name_of_xml)


        if len(regions) != 0:  
            if regions != []:
                if regions != [[]]:
                    if state == 'updated' or state == 'tagged':
                        #save label in a list 
                        for j in regions:
                            class_label_list.append(str(j['cls']))

                        #copy file to annotation_dir
                        shutil.copy2(file_path, all_img_dir)

                        #generate xml directly to annotaion_dir
                        annotation = ET.Element('annotation')
                        ET.SubElement(annotation, 'folder').text = str(folder_name)
                        ET.SubElement(annotation, 'filename').text = str(name_of_img)
                        ET.SubElement(annotation, 'path').text = str(img_file_path)


                        size = ET.SubElement(annotation, 'size')
                        ET.SubElement(size, 'width').text = str(width)
                        ET.SubElement(size, 'height').text = str(height)
                        ET.SubElement(size, 'depth').text = str(depth)

                        ET.SubElement(annotation, 'segmented').text = '0'
                
                        for j in curr_annotation:

                            ob = ET.SubElement(annotation, 'object')
                            ET.SubElement(ob, 'name').text = str(j[4])
                            ET.SubElement(ob, 'pose').text = 'Unspecified'
                            ET.SubElement(ob, 'truncated').text = '1'
                            ET.SubElement(ob, 'difficult').text = '0'

                            bbox = ET.SubElement(ob, 'bndbox')
                            ET.SubElement(bbox, 'xmin').text = str(j[0])
                            ET.SubElement(bbox, 'ymin').text = str(j[1])
                            ET.SubElement(bbox, 'xmax').text = str(j[2])
                            ET.SubElement(bbox, 'ymax').text = str(j[3])
                    
                            tree = ET.ElementTree(annotation) 
                            #write xml in annotation folder
                            with open(all_xml_dir, "wb") as files : 
                                tree.write(files)


    #split 80/20 in annotation dir to its respective folders
    from sklearn.model_selection import train_test_split


    image_files = os.listdir(all_img_dir)

    #image_names = [name.replace(".jpg","") for name in image_files]

    test_names, train_names = train_test_split(image_files, test_size=0.2)

    for test_n in test_names:
        xml_name = str(test_n).split(".")[0] + ".xml"

        source = os.path.join(all_img_dir,test_n)
        dest = os.path.join(crop_test_img_dir,test_n)
        shutil.copy2(source,dest)

        source = os.path.join(all_xml_dir,xml_name)
        dest = os.path.join(crop_test_xml_dir,xml_name)
        shutil.copy2(source,dest)


    for train_n in train_names:
        xml_name = str(train_n).split(".")[0] + ".xml"

        source = os.path.join(all_img_dir,train_n)
        dest = os.path.join(crop_train_img_dir,train_n)
        shutil.copy2(source,dest)

        source = os.path.join(all_xml_dir,xml_name)
        dest = os.path.join(crop_train_xml_dir,xml_name)
        shutil.copy2(source,dest)

    
    #remove all_img_dir (dont need this no more)
    all_img_dir = os.path.join(temp_dir,'all_img')
    if not os.path.exists(all_img_dir):
        os.makedirs(all_img_dir)
    else:
        shutil.rmtree(all_img_dir, ignore_errors=True)

    #remove all_xml_dir (dont need this no more)
    all_xml_dir = os.path.join(temp_dir,'all_xml')
    if not os.path.exists(all_xml_dir):
        os.makedirs(all_xml_dir)
    else:
        shutil.rmtree(all_xml_dir, ignore_errors=True)

    
    #make classes.txt in classes_dir folder
    path_to_class_txt = os.path.join(classes_dir,'classes.txt')
    with open(path_to_class_txt, 'w') as f:
        for item in class_label_list:
            f.write("%s\n" % item)

    #generate tf_records by calling custom monk ai script 

    import sys
    sys.path.append("livis/Monk_Object_Detection/13_tf_obj_2/lib/")
    sys.path.append("livis/Monk_Object_Detection/13_tf_obj_2/lib/models/")
    sys.path.append("livis/Monk_Object_Detection/13_tf_obj_2/lib/models/research/")
    sys.path.append("livis/Monk_Object_Detection/13_tf_obj_2/lib/models/research/object_detection/")
    sys.path.append("livis/Monk_Object_Detection/13_tf_obj_2/lib/models/research/object_detection/models/")
    from livis.Monk_Object_Detection.tf_obj_2.lib.train_detector import Detector
    gtf = Detector()

    gtf.set_train_dataset(crop_train_img_dir, crop_train_xml_dir, path_to_class_txt, batch_size=batch_size)
    gtf.set_val_dataset(crop_test_img_dir, crop_test_xml_dir)


    gtf.create_tfrecord(data_output_dir=tf_records_dir)
    if model_type is None:
        model_type = "ssd_mobilenet_v2_fpnlite_640"
    gtf.set_model_params(model_name=model_type)
    gtf.set_hyper_params(num_train_steps=num_steps, lr=lr)
    gtf.export_params(output_directory=export_dir)

    os.system("python livis/Monk_Object_Detection/tf_obj_2/lib/train.py")






    message = l
    status_code = 200  
    return message,status_code



