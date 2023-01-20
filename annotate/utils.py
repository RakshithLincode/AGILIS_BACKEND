#from common.utils import MongoHelper, run_in_background
from common.utils import MongoHelper

from annotate.constants import *
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
########## manual annotation api ##########

#ui send jigid and fileid throw nxt fileid after iteration over db
def next_img_util(data):

	message = None
	status_code = None

	jig_id = data.GET['jig_id']
	if jig_id is None:
		message = "jigId not provided"
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
			message = "Jig not found in Jig collection"
			status_code = 404
			return message,status_code

	except Exception as e:
		message = "Invalid jigID"
		status_code = 400
		return message,status_code

	file_id = data.GET['file_id']
	if file_id is None:
		message = "FileId not provided"
		status_code = 400
		return message,status_code

		
	try:   
		mp = MongoHelper().getCollection(str(dataset['_id']))
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code
	
	try:
		dataset = mp.find_one({'_id' : ObjectId(str(file_id))})
		if dataset is None:
			message = "File not found in jig collection"
			status_code = 404
			return message,status_code
		
	except Exception as e:
		message = "Invalid FileID"
		status_code = 400
		return message,status_code

	currId = str(dataset['_id'])

	p = [i for i in mp.find()]
	idx = 0
	for i in p:
		print(i['_id'])
		if str(i['_id']) == str(file_id):
			break
		idx += 1

	#print(p)

	try: 
		q = p[idx+1] 
	except:
		q = p[idx]

	message = q
	status_code = 200
	return message,status_code

#ui send jigid and fileid throw prev fileid after iteration over db
def prev_img_util(data):

	message = None
	status_code = None

	jig_id = data.GET['jig_id']
	if jig_id is None:
		message ="jigId not provided"
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
			message = "Jig not found in Jig collection"
			status_code = 404
			return message,status_code

	except Exception as e:
		message = "Invalid jigID"
		status_code = 400
		return message,status_code

	file_id = data.GET['file_id']
	if file_id is None:
		message = "FileId not provided"
		status_code = 400
		return message,status_code

		
	try:   
		mp = MongoHelper().getCollection(str(dataset['_id']))
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code
	
	try:
		dataset = mp.find_one({'_id' : ObjectId(str(file_id))})
		if dataset is None:
			message = "File not found in jig collection"
			status_code = 404
			return message,status_code
		
	except Exception as e:
		message = "Invalid FileID"
		status_code = 400
		return message,status_code

	currId = str(dataset['_id'])

	p = [i for i in mp.find()]
	idx = 0
	for i in p:
		print(i['_id'])
		if str(i['_id']) == str(file_id):
			break
		idx += 1

	#print(p)

	try: 
		if idx == 0:
			q = q = p[idx]
		else:
			q = p[idx-1] 
	except:
		q = p[idx]

	message = q
	status_code = 200
	return message,status_code

#ui send jigid and fileid throw current fileid after iteration over db
def get_img_util(data):

	message = None
	status_code = None

	jig_id = data.GET['jig_id']
	if jig_id is None:
		message = "jigId not provided"
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
			message = "Jig not found in Jig collection"
			status_code = 404
			return message,status_code

	except Exception as e:
		message = "Invalid jigID"
		status_code = 400
		return message,status_code

	file_id = data.GET['file_id']
	if file_id is None:
		message = "FileId not provided"
		status_code = 400
		return message,status_code

		
	try:   
		mp = MongoHelper().getCollection(str(dataset['_id']))
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code
	
	try:
		dataset = mp.find_one({'_id' : ObjectId(str(file_id))})
		if dataset is None:
			message = "File not found in jig collection"
			status_code = 404
			return message,status_code
		
	except Exception as e:
		message = "Invalid FileID"
		status_code = 400
		return message,status_code
	
	#p = [i for i in mp.find()]

	message = dataset
	status_code = 200

	return message,status_code


def delete_img_util(data):

	message = None
	status_code = None

	jig_id = data['jig_id']
	if jig_id is None:
		message = "jigId not provided"
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
			message = "Jig not found in Jig collection"
			status_code = 404
			return message,status_code

	except Exception as e:
		message = "Invalid jigID"
		status_code = 400
		return message,status_code


	file_id = data['file_id']
	if file_id is None:
		message = "FileId not provided"
		status_code = 400
		return message,status_code

		
	try:   
		mp = MongoHelper().getCollection(str(dataset['_id']))
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code
	
	try:
		dataset = mp.find_one({'_id' : ObjectId(str(file_id))})
		if dataset is None:
			message = "File not found in jig collection"
			status_code = 404
			return message,status_code
		
	except Exception as e:
		message = "Invalid FileID"
		status_code = 400
		return message,status_code
	
	#p = [i for i in mp.find()]
	file_path = dataset['file_path']
	if os.path.exists(file_path): 
		os.remove(file_path)

	#_id = dataset['_id']

	try:

		import pymongo
		myclient = pymongo.MongoClient("mongodb://localhost:27017/")

		mydb = myclient["LIVIS"]
		mycol = mydb[str(jig_id)]

		result = mycol.delete_one({'_id': ObjectId(file_id)})

	except Exception as e:
		print(e)
		message = "error while deleting from db"
		status_code = 500

	message = dataset
	status_code = 200

	return message,status_code

#same as fetch all jig -- not sure where its used
def get_dataset_list_util(skip=0, limit=100):

	try:
		mp = MongoHelper().getCollection(JIG_COLLECTION)
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code
		#error connecting to db

	try:   
		datasets = mp.find().skip(skip).limit(limit)
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code
		#error connecting to db

	datasets = [i for i in datasets]

	message = datasets
	status_code = 200

	return message,status_code
	


"""
#upload zip functionality - disable in schneider (send same statistics when all capture is done)
def create_dataset_util(data):



	#get data from form to variables

	# error handle zip file
	file = data.FILES.get('myfile')
	if file is None:
		message = "No zip file provided"
		status_code = 400
		return message,status_code

	try:
		ext=str(file).split('.')[-1]
		if ext != 'zip':
			message = "Not a zip file"
			status_code = 415
			return message,status_code
	except:
		message = "Not a zip file"
		status_code = 415
		return message,status_code
	
	# error handle db connectivity
	try:
		mp = MongoHelper().getCollection(JIG_COLLECTION)
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code
		#error connecting to db


	# error handle part_id
	part_id =  data.POST.get('part_id')
	if part_id is None:
		message = "PartId not provided"
		status_code = 400
		return message,status_code


	try:
		dataset = mp.find_one({'_id' : ObjectId(part_id.replace('"',''))})
		if dataset is None:
			message = "Part not found in Parts collection"
			status_code = 404
			return message,status_code

	except Exception as e:
		message = "Invalid partID"
		status_code = 400
		return message,status_code

	

	try:
		part_number =  data.POST.get('part_number')
	except Exception as e:
		message = "Part number not provided"
		status_code = 400
		return message,status_code

	
	try: 
		line_number =  data.POST.get('line_number')
	except Exception as e:
		message = "Line number not provided"
		status_code = 400
		return message,status_code

	try:  
		factory_code =  data.POST.get('factory_code')
	except Exception as e:
		message = "Factory code not provided"
		status_code = 400
		return message,status_code
	

	names = []

	#extract
	with ZipFile(file) as zip_file:
		names = zip_file.namelist()
		for name in names:
			if 'jpg' in name or 'png' in name:    
				 # copy file (taken from zipfile's extract)
				filename = os.path.basename(name)
				source = zip_file.open(name)
				target = open(os.path.join(settings.TRAIN_DATA_STATIC, filename), "wb")
				
				with source, target:
					shutil.copyfileobj(source, target)
					
				#zip_file.extract(name, path=setttings.TRAIN_DATA_STATIC)
	
	#remove all non image from list
	new1 = [x for x in names if "." in str(x)]
	new = [x for x in new1 if x.split('.')[1] == 'png' or x.split('.')[1] == 'jpg']

	if new == []:
		#no images found in zip
		message = "Could not find any images in zip"
		status_code = 404
		return message,status_code


	## add meta info in parts collection -- might change according to customer --

	def add_metadata(part_number,line_number,factory_code,mp):

		collection_obj = {
			'part_number':part_number,
			'line_number':line_number,
			'factory_code':factory_code
		}

		mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  collection_obj})


	#add_metadata(part_number,line_number,factory_code,mp)

	
	_id = str(dataset['_id'])
	mp = MongoHelper().getCollection(str(dataset['_id']))
	p = [i for i in mp.find()]

	for i in new:

		head_tail = os.path.split(i)

		other_path = head_tail[0]
		name_of_img = head_tail[1]

		collection_obj = {
		'file_path' : os.path.join(settings.TRAIN_DATA_STATIC,name_of_img),
		'file_url' : "http://164.52.194.78:3306/" + name_of_img,
		'state' : 'untagged',
		'regions' : [],
		'regions_history' : [],
		'classifier_label' : "",
		'classifier_label_history' : [],
		'annotator' : ''
		}
		
		mp.insert(collection_obj)

	mp = MongoHelper().getCollection(str(dataset['_id']))
	p = [i for i in mp.find()]

	url_lst = []
	for i in p:
		semi_path = (str(i['file_path']).replace(settings.TRAIN_DATA_STATIC,""))
		semi_path = semi_path.replace('\\','')
		url_lst.append( "http://localhost"  + semi_path )

	def get_metrics(mp):
		data = {}
		data['total_images'] = len(p)
		data['total_labeled_images'] = 0
		data['total_unlabeled_images'] = 0
		for i in p:
			if i['state'] == 'tagged' or i['state'] == 'semi-tagged':
				data['total_labeled_images'] += 1
			elif i['state'] == 'untagged':
				data['total_unlabeled_images'] += 1

		return data

	resp = get_metrics(mp)
	


	message = resp
	status_code = 200

	return message,status_code
"""

# continously fetched from ui 
def get_data_for_histogram_util(data):

	try:
		mp = MongoHelper().getCollection(JIG_COLLECTION)
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code

	# error handle jig_id
	jig_id = data.GET['jig_id']
	if jig_id is None:
		message = "jig_id not provided"
		status_code = 400
		return message,status_code

	try:
		dataset = mp.find_one({'_id' : ObjectId(jig_id.replace('"',''))})
		if dataset is None:
			message = "jig not found in jig collection"
			status_code = 404
			return message,status_code

	except Exception as e:
		message = "Invalid jigid"
		status_code = 400
		return message,status_code

	#p = [i for i in mp.find()]

	_id = str(dataset['_id'])
	mp = MongoHelper().getCollection(str(dataset['_id']))
	p = [i for i in mp.find()]

	def get_metrics(mp):
		data = {}
		data['total_images'] = len(p)
		data['total_labeled_images'] = 0
		data['total_unlabeled_images'] = 0
		for i in p:
			if i['state'] == 'tagged' or i['state'] == 'semi-tagged':
				data['total_labeled_images'] += 1
			elif i['state'] == 'untagged':
				data['total_unlabeled_images'] += 1

		return data

	resp = get_metrics(mp)

	message = resp
	status_code = 200

	return message,status_code



#fetch all file_url of crops to display in ui 
def fetch_image_url_util(data):
	"""
	usage : GET REQUEST JSON
	
	"jig_id": "5f11996da729dc68314aeda2"

	What it does:   returns list of urls for web to render in FE 
	"""

	#get data from json to variables
	try:
		mp = MongoHelper().getCollection(JIG_COLLECTION)
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code

	# error handle jig_id
	jig_id = data.GET['jig_id']
	if jig_id is None:
		message = "jig_id not provided"
		status_code = 400
		return message,status_code

	try:
		dataset = mp.find_one({'_id' : ObjectId(jig_id.replace('"',''))})
		if dataset is None:
			message = "jig not found in jig collection"
			status_code = 404
			return message,status_code

	except Exception as e:
		message = "Invalid jig_id"
		status_code = 400
		return message,status_code


	_id = str(dataset['_id'])

	mp = MongoHelper().getCollection(str(dataset['_id']))
	p = [i for i in mp.find()]
	url_lst = []
	for i in p:
		url_lst.append(i['file_url'])
		#print(str(i['file_path'])-str(IMAGE_DATASET_SAVE_PATH))

	message = url_lst
	status_code = 200

	return message,status_code


#store individual crop annotaion

def submit_annotations_util(data):
	"""
	usage : POST REQUEST JSON
	{
	"src": "http://164.52.194.78:3306/sample/1M07AE_1_33.png",
	"classifier_label" : "",
	"id": "5f5722098692f6ec4c1d4b06",
	"jig_id": "5f42b5a90b72e674c03691d5",
	"pixelSize": {
		"w": 644,
		"h": 882
	},
	"regions": [{
		"type": "box",
		"x": 0.3144927536231884,
		"y": 0.23208333333333334,
		"w": 0.4646376811594203,
		"h": 0.28,
		"highlighted": false,
		"editingLabels": false,
		"color": "#4caf50",
		"cls": "test1",
		"id": "9846203889465652",
		"tags": ["tag1"]
	}, {
		"type": "box",
		"x": 0.6066666666666666,
		"y": 0.10912037037037037,
		"w": 0.38550724637681166,
		"h": 0.04444444444444445,
		"highlighted": true,
		"editingLabels": false,
		"color": "#2196f3",
		"cls": "Hole",
		"id": "6363653630439714",
		"tags": ["tag1"]
	}]
	}

	What it does:   for that specific annotation image
					adds annotation [[x0,y0,x1,y1,label]] or [[label]]
					change status from untagged to semitagged if image is annotated
					adds annotator field info
					returns data of that record after being updated in json message format
	"""


	#get data from json to variables
	
	try:
		jig_id = data.get('jig_id')
	except:
		message = "jig_id not provided"
		status_code = 400
		return message,status_code

	try:
		file_id = data.get('id')
	except:
		message = "FileId not provided"
		status_code = 400
		return message,status_code

	try:
		pixelSize = data.get('pixelSize')
	except:
		message = "height and width not provided"
		status_code = 400
		return message,status_code

	try:
		regions = data.get('regions')
	except:
		message = "regions not provided"
		status_code = 400
		return message,status_code


	classifier_label = data.get('classifier_label')
	
	#wandh = pixelSize["pixelSize"]
	width = pixelSize["w"]
	height = pixelSize["h"]

	cords = []

	for i in regions:
		x = i["x"]
		y = i["y"]
		w = i["w"]
		h = i["h"]

		x0 = x * width
		y0 = y * height
		x1 = w * width
		y1 = y * height

		label = i["cls"]

		cords.append([x0,y0,x1,y1,label])


	mp = MongoHelper().getCollection(JIG_COLLECTION)
	dataset = mp.find_one({'_id' : ObjectId(jig_id)})
	mp = MongoHelper().getCollection(str(dataset['_id']))
	dataset = mp.find_one({'_id' : ObjectId(file_id)})


	#change annotation if modified
	prev_classifier_label = dataset['classifier_label']
	prev_region = dataset['regions']
	classifier_label_history = ['classifier_label_history']
	regions_history = ['regions_history']


	#push to history if prev exists
	#if prev_classifier_label != "" and classifier_label != "":
	#    classifier_label_history.append(prev_classifier_label)

	#if prev_region != [] and regions != []:

	#change status
	if classifier_label == "" and regions == [] :
		state = 'untagged'
	else:
		state = 'tagged'


	#prepare new data to push
	data = {
		'state': state,
		'classifier_label' : classifier_label,
		'regions' : regions,
		'classifier_label_history' : classifier_label_history,
		'regions_history' : regions_history
	}

	#goto that image in db
	mp = MongoHelper().getCollection(JIG_COLLECTION)
	dataset = mp.find_one({'_id' : ObjectId(jig_id)})
	mp = MongoHelper().getCollection(str(dataset['_id']))
	dataset = mp.find_one({'_id' : ObjectId(file_id)})

	#print(ObjectId(dataset['_id']))
	mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  data})

	message = data
	status_code = 200

	return message,status_code

#getdata
def fetch_data_util(data):
	"""
	usage : GET REQUEST JSON
	{
	"jig_id": "5f11996da729dc68314aeda2"
	}

	What it does:   fetch all list of images along with its annotated data after annotation 
						from mongo db for UI to display and check annotation
					
					returns json containing list of images and its annotation data.
	"""

	try:
		mp = MongoHelper().getCollection(JIG_COLLECTION)
	except:
		message = "Cannot connect to db"
		status_code = 500
		return None,None,None,message,status_code

	# error handle jig_id
	jig_id = data.GET['jig_id']
	current  = None
	limit_to = None

	try:
		current = data.GET['current']
	except:
		pass
	try:
		limit_to = data.GET['limit']
	except:
		pass

	if current is not None:
		current = int(current)
	if limit_to is not None:
		limit_to = int(limit_to)

	if jig_id is None:
		message = "jig_id not provided"
		status_code = 400
		return None,None,None,message,status_code

	try:
		dataset = mp.find_one({'_id' : ObjectId(jig_id.replace('"',''))})

		if dataset is None:
			message = "jig not found in jig collection"
			status_code = 404
			return None,None,None,message,status_code

	except Exception as e:
		message = "Invalid jig_id"
		status_code = 400
		return None,None,None,message,status_code

	_id = str(dataset['_id'])

	mp = MongoHelper().getCollection(str(dataset['_id']))

	if current is None and limit_to is None:
		p = [i for i in mp.find()]
	else:
		
		if current == 1:
			p = [i for i in mp.find().skip(0).limit(limit_to)]
		else:
			skip_from = ((current * limit_to)-limit_to)
			p = [i for i in mp.find().skip(skip_from).limit(limit_to)]

	q = [i for i in mp.find()]
	total = len(q)
	message = p
	status_code = 200

	return total,current,limit_to,message,status_code

"""
def check_annotations_util(data):
	
	usage : POST REQUEST JSON
	{
	"part_id": "5f11996da729dc68314aeda2",
	"file_id":"5f11996ea729dc68314aeda3",
	"classifier_annotation": [["k705"]],
	"detector_annotation" : [[20,20,30,30,"k705"],[10,10,40,40,"k706"]]
	}

	What it does:   corrects wrong annotation done if any
					change status from semi tagged to updated
					push annotation to history when there is a change/correction done and add new annotation cords or label to annotation array
					returns json containing that corrected and status updated record
	
	#get data from json to variables

	part_id = data.get('part_id')
	if part_id is None:
		message = "PartId not provided"
		status_code = 400
		return message,status_code


	file_id = data.get('file_id')
	if file_id is None:
		message = "FileId not provided"
		status_code = 400
		return message,status_code



	classifier_annotation = data.get('classifier_annotation')
	detector_annotation = data.get('detector_annotation')
	#goto that image in collection


	try:
		mp = MongoHelper().getCollection(PARTS_COLLECTION)
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code

	dataset = mp.find_one({'_id' : ObjectId(part_id)})
	mp = MongoHelper().getCollection(str(dataset['_id']))
	dataset = mp.find_one({'_id' : ObjectId(file_id)})


	#change annotation if modified
	prev_classifier_annotation = dataset['classifier_annotation']
	prev_classifier_annotator = dataset['annotator']
	prev_detector_annotation = dataset['detector_annotation']
	prev_detector_annotator = dataset['annotator']
	prev_classifier_history = dataset['classifier_annotation_history']
	prev_detector_history = dataset['detector_annotation_history']

	#if prev_classifier_annotation != []:
	#    prev_classifier_annotation.append(prev_classifier_annotation)
	#if prev_detector_annotation != []:
	#    prev_detector_annotation.append(prev_detector_annotation)


	#if len(prev_annotation) != len(annotation) or prev_annotation!=annotation:

	previous_classifier_obj = {"classifier_annotation":prev_classifier_annotation}
	previous_detector_obj = {"detector_annotation":prev_detector_annotation}

	prev_classifier_history.append(previous_classifier_obj)
	prev_detector_history.append(previous_detector_obj)

	state = 'updated'

	data = {
		'state' : state,
		'classifier_annotation' : classifier_annotation,
		'detector_annotation' : detector_annotation,
		'classifier_annotation_history' : prev_classifier_history,
		'detector_annotation_history' : prev_detector_history
	}

	#print(ObjectId(dataset['_id']))
	mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  data})
	dataset = mp.find_one({'_id' : ObjectId(file_id)})
	
	#else just update status
	#else:

	#    state = 'updated'

	#    data = {
	#        'state': state
	#    }

		#print(ObjectId(dataset['_id']))
		#mp.update({'_id' : ObjectId(dataset['_id'])}, {'$set' :  data})
		#dataset = mp.find_one({'_id' : ObjectId(file_id)})
	
	message = dataset
	status_code = 200
	return message,status_code
"""

def export_data_utils(data):
	#from common.utils import GetLabelData
	#data = GetLabelData("5f5229175f70f967162b7c26").get_metrics()
	#return data,200
	#from common.utils import ExperimentDataset
	#traindf,testdf = ExperimentDataset("5f5229175f70f967162b7c26","5f58769fcf00e198c3a3c0c2").to_csv("classification")
	#print(traindf)
	#print(testdf)

	'''
	usage : POST REQUEST JSON
	{
	"jig_id": "5f11996da729dc68314aeda2",
	"output_type": "csv" or "xml"
	}

	What it does:   exports the annotation to xml format (detector)
					exports the annotation to csv fomrat (classifier)

					store xml beside image in same directory
					store one csv file in seperate directory beside image directory

					return xml file path or csv file path
	'''

	#get data from json to variables
	jig_id = data.get('jig_id')
	if jig_id is None:
		message = "jig_id not provided"
		status_code = 400
		return message,status_code

	output_type = data.get('output_type')
	if output_type is None:
		message = "output_type not provided"
		status_code = 400
		return message,status_code

	try:
		mp = MongoHelper().getCollection(JIG_COLLECTION)
	except:
		message = "Cannot connect to db"
		status_code = 500
		return message,status_code

	dataset = mp.find_one({'_id' : ObjectId(jig_id)})
	_id = str(dataset['_id'])

	mp = MongoHelper().getCollection(str(dataset['_id']))
	p = [i for i in mp.find()]

	if len(p) == 0:
		message = "no records found in specified collection"
		status_code = 400
		return message,status_code

	if output_type == 'xml':
		for i in p:

			img_file_path = i['file_path']
			curr_annotation = i['detector_annotation']
			state = i['state']
			
			head_tail = os.path.split(img_file_path)

			xml_file_path = head_tail[0]
			name_of_img = head_tail[1]
			name_of_xml = str(name_of_img.split('.')[0]) +".xml"
			full_path_of_xml = os.path.join(xml_file_path,name_of_xml)

			folder_name = os.path.basename(xml_file_path)
			im = cv2.imread(img_file_path)
			height,width,depth = im.shape

			if len(curr_annotation) != 0:  
				if curr_annotation != []:
					if curr_annotation != [[]]:
						if state == 'updated' or state == 'tagged':

							annotation = ET.Element('annotation')
							ET.SubElement(annotation, 'folder').text = str(folder_name)
							ET.SubElement(annotation, 'filename').text = str(name_of_img)
							ET.SubElement(annotation, 'path').text = str(img_file_path)


							size = ET.SubElement(annotation, 'size')
							ET.SubElement(size, 'width').text = str(width)
							ET.SubElement(size, 'height').text = str(height)
							ET.SubElement(size, 'depth').text = str(depth)

							ET.SubElement(annotation, 'segmented').text = '0'
					
							for i in curr_annotation:

								ob = ET.SubElement(annotation, 'object')
								ET.SubElement(ob, 'name').text = str(i[4])
								ET.SubElement(ob, 'pose').text = 'Unspecified'
								ET.SubElement(ob, 'truncated').text = '1'
								ET.SubElement(ob, 'difficult').text = '0'

								bbox = ET.SubElement(ob, 'bndbox')
								ET.SubElement(bbox, 'xmin').text = str(i[0])
								ET.SubElement(bbox, 'ymin').text = str(i[1])
								ET.SubElement(bbox, 'xmax').text = str(i[2])
								ET.SubElement(bbox, 'ymax').text = str(i[3])

						
								tree = ET.ElementTree(annotation) 

								with open(full_path_of_xml, "wb") as files : 
									tree.write(files)
		
		return xml_file_path

	if output_type == 'csv':
		list_of_rows = []

		for i in p:
			img_file_path = i['file_path']
			curr_annotation = i['classifier_annotation']
			state = i['state']
			
			head_tail = os.path.split(img_file_path)

			abs_path = head_tail[0]
			name_of_img = head_tail[1]

			out_of_abs = os.path.split(abs_path)
			base_path = out_of_abs[1]
			out_of_abs = out_of_abs[0]


			export_path = os.path.join(out_of_abs,"export")
			annotation_path = os.path.join(out_of_abs,"annotations")
			os.makedirs(export_path, exist_ok=True)
			os.makedirs(annotation_path, exist_ok=True)
			
			csv_file_path = os.path.join(export_path,base_path+".csv")

			
			if curr_annotation != []:
				if curr_annotation != [[]]:
					if  state == 'updated' or 'tagged':
						if len(curr_annotation) == 1:
							
							curr_annotation = curr_annotation[0][0]
							list_of_rows.append([img_file_path,curr_annotation])
			
		
		if len(list_of_rows)>0:

			fields = ["file_path","label"]
			with open(csv_file_path,'w')as csvfile:
				csvwriter = csv.writer(csvfile)  
 
				csvwriter.writerow(fields)
				csvwriter.writerows(list_of_rows)

		message = export_path
		status_code = 200
		return message,status_code
		#return export_path
	

