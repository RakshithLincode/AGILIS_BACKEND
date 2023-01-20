from common.utils import *
#from django.utils import timezone
import datetime
from bson import ObjectId
from toyoda.kanban import static_kanban
from livis.settings import *
from plan.utils import *
import cv2
from accounts.utils import get_user_account_util


def set_kanban_on_redis(workstation_name, short_number):
    cc = CacheHelper()
    kanban_key = RedisKeyBuilderServer(workstation_name).get_key(0, 'kanban')
    cc.set_json({kanban_key : static_kanban[short_number]})


def start_toyoda_process(data):
    
    part_number = data.get('part_number',None)
    user_id = data.get('user_id')
    #model_number = data.get('model_number')
    #short_number = data.get('short_number')
    part_description = data.get('part_description')
    #plan = get_todays_planned_production_util(short_number)
    #current_production_count = plan['planned_production_count']
    workstation_id = data.get('workstation_id')
    user_details = get_user_account_util(user_id)
    print("user_details::::",user_details)
    role_name = user_details['role_name']
    print("role_name::::",role_name,type(role_name))
    user = { "user_id": user_id,
                "role": user_details['role_name'],
                "name": (user_details['first_name']+" "+user_details['last_name'])
            }
    print("user:::: ",user)
    createdAt = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    mp = MongoHelper().getCollection('inspection_data')
    obj = {
        'part_number' : part_number,
        'user' : user,
        #"model_number" : model_number,
        "workstation_id" : workstation_id,
        #"short_number" : short_number,
        "part_description" : part_description,
        #"plan" : plan,
        "status" : 'started',
        'createdAt' : createdAt,
        #'current_production_count' : current_production_count,
        'is_manual' : True
    }
    #set_kanban_on_redis(workstation_id, short_number) 
    _id = mp.insert(obj)
    workstation_info = RedisKeyBuilderServer(workstation_id).workstation_info
    cc = CacheHelper()
    curr_inspection_key = RedisKeyBuilderServer(workstation_id).get_key(0, 'curr-inspection-id')
    if curr_inspection_key:
        cc.set_json({curr_inspection_key : str(_id)}) 
        resp = mp.find_one({"_id" : _id})
        if resp:
            return resp
        else:
            return {}
    else:
        return {}

def start_schneider_process(data):
    jig_id = data.get('jig_id',None)
    user_id = data.get('user_id')
    #model_number = data.get('model_number')
    #short_number = data.get('short_number')
    #part_description = data.get('part_description')
    #plan = get_todays_planned_production_util(short_number)
    #current_production_count = plan['planned_production_count']
    #workstation_id = data.get('workstation_id')
    user_details = get_user_account_util(user_id)
    print("user_details::::",user_details)
    role_name = user_details['role_name']
    print("role_name::::",role_name,type(role_name))
    user = { "user_id": user_id,
                "role": user_details['role_name'],
                "name": (user_details['first_name']+" "+user_details['last_name'])
            }
    print("user:::: ",user)
    createdAt = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    mp = MongoHelper().getCollection('inspection_data')
    obj = {
        'jig_id' : jig_id,
        'user' : user,
        #"model_number" : model_number,
        #"workstation_id" : workstation_id,
        #"short_number" : short_number,
        #"part_description" : part_description,
        #"plan" : plan,
        "status" : 'started',
        'createdAt' : createdAt,
        #'current_production_count' : current_production_count,
        'is_manual_pass' : False
    }
    #set_kanban_on_redis(workstation_id, short_number) 
    _id = mp.insert(obj)
    #workstation_info = RedisKeyBuilderServer(workstation_id).workstation_info
    #cc = CacheHelper()
    #curr_inspection_key = RedisKeyBuilderServer(workstation_id).get_key(0, 'curr-inspection-id')
    #if curr_inspection_key:
    #    cc.set_json({curr_inspection_key : str(_id)}) 

    resp = mp.find_one({"_id" : _id})
    if resp:
        return resp
    else:
        return {}

     
    
def end_toyoda_process(data):
    mp = MongoHelper().getCollection('inspection_data')
    _id = data.get('process_id')
    pr = mp.find_one({"_id" : ObjectId(_id)})
    if pr:
        pr['completedAt'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        completedAt = datetime.datetime.strptime(pr['completedAt'], '%Y-%m-%d %H:%M:%S')
        createdAt = datetime.datetime.strptime(pr['createdAt'], '%Y-%m-%d %H:%M:%S')
        duration = completedAt - createdAt
        pr['duration'] = str(duration)
        pr['status'] = 'completed'
        #pr['plan'] = get_todays_planned_production_util(pr['short_number'])
        inspection_attributes = MongoHelper().getCollection(str(_id))
        total_accepted_parts = inspection_attributes.find({"isAccepted" : True}).count()
        total_parts = inspection_attributes.find().count()
        total_rejected_parts = total_parts - total_accepted_parts
        pr['total_accepted_parts'] = total_accepted_parts
        pr['total_rejected_parts'] = total_rejected_parts
        pr['total_parts'] = total_parts
        mp.update({'_id' : pr['_id']}, {'$set' : pr})
        resp = mp.find_one({"_id" : ObjectId(_id)})
        if resp:
            return resp
        else:
            return {}
    else:
        return {}

def end_schneider_process(data):
    mp = MongoHelper().getCollection('inspection_data')
    _id = data.get('process_id')
    pr = mp.find_one({"_id" : ObjectId(_id)})
    if pr:
        pr['completedAt'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        completedAt = datetime.datetime.strptime(pr['completedAt'], '%Y-%m-%d %H:%M:%S')
        createdAt = datetime.datetime.strptime(pr['createdAt'], '%Y-%m-%d %H:%M:%S')
        duration = completedAt - createdAt
        pr['duration'] = str(duration)
        pr['status'] = 'completed'
        #pr['plan'] = get_todays_planned_production_util(pr['short_number'])
        inspection_attributes = MongoHelper().getCollection(str(_id))
        total_accepted_parts = inspection_attributes.find({"isAccepted" : True}).count()
        total_parts = inspection_attributes.find().count()
        total_rejected_parts = total_parts - total_accepted_parts
        pr['total_accepted_parts'] = total_accepted_parts
        pr['total_rejected_parts'] = total_rejected_parts
        pr['total_parts'] = total_parts
        mp.update({'_id' : pr['_id']}, {'$set' : pr})
        resp = mp.find_one({"_id" : ObjectId(_id)})
        if resp:
            return resp
        else:
            return {}
    else:
        return {}

def get_toyoda_running_process(worksatation_id):
    mp = MongoHelper().getCollection('inspection_data')
    prs = mp.find({"$and": 
                [{"$or":
                [{'status' : 'started'},
                {'status' : 'planned_production_not_achieved'},
                {'status' : 'qc_inspection_failed'}]},
                {"workstation_id": worksatation_id}]
                }).sort([( '$natural', -1 )] )
    if prs.count() > 0:
        pr = prs[0]
        #plan_update = pr['plan']
        #plan_update['planned_production_count'] = pr['current_production_count']
        #print("plan_update:::::::",plan_update)
        #pr['plan'] = plan_update
        cc = CacheHelper()
        #production_count = pr['current_production_count']
        production_count_key = RedisKeyBuilderServer(worksatation_id).get_key(0, 'production_count_key')
        #print("production_count:::::::",production_count)
        cc.set_json({production_count_key : production_count}) 
        return pr
    else:
        return {}


def update_inspection_manual(data):
    mp = MongoHelper().getCollection('inspection_data')
    qc_status = data.get('qc_status')
    #part_weight = data.get('part_weight')
    qc_remark = data.get('qc_remark')
    inspection_type = data.get('inspection_type')
    _id = data.get('process_id')
    upd_obj = {
        "qc_status" : qc_status,
        #"part_weight" : part_weight,
        "qc_remark" : qc_remark,
        "inspection_type" : inspection_type
    }
    pr = mp.find_one({'_id' : ObjectId(_id)})
    if 'manual_inspection_result' in pr:
        pr['manual_inspection_result'].append(upd_obj)
    else:
        pr['manual_inspection_result'] = [upd_obj]
    mp.update({'_id' : pr['_id']}, {'$set' : pr})
    resp = mp.find_one({"_id" : pr['_id']})
    if resp:
        return resp
    else:
        return {}


def get_camera_feed_urls(workstation_id):
    feed_urls = {}
    workstation_info = RedisKeyBuilderServer(workstation_id).workstation_info
    for camera_info in workstation_info['camera_config']['cameras']:
        url = "http://127.0.0.1:8000/livis/v1/toyoda/stream/{}/{}/".format(workstation_id,camera_info['camera_id'])
        # url = "http://0.0.0.0:8000/livis/v1/toyoda/stream1/{}/{}/".format(workstation_id,camera_info['camera_id'])
        feed_urls[camera_info['camera_name']] = url
    if feed_urls:
        return feed_urls
    else:
        return {}


def rescan_util(data):
    mp = MongoHelper().getCollection('inspection_data')
    process_id = data.get('process_id')
    inspection_id = data.get('inspection_id')
    defects_list = data.get('defects_list')
    features_list = data.get('features_list')
    serial_number = data.get('serial_number')
    isAccepted = data.get('isAccepted')
    taco_fail = data.get('taco_fail')
    taco_pass = data.get('taco_pass')
    timestamp = data.get('timestamp')
    pr = mp.find_one({"_id" : ObjectId(process_id)})
    if pr:
        inspection_attributes = MongoHelper().getCollection(str(process_id))
        inspection = inspection_attributes.find_one({"_id" : ObjectId(inspection_id)})
        if inspection:
            inspection['defects_list'] = defects_list
            inspection['features_list'] = features_list
            inspection['serial_number'] = serial_number
            inspection['isAccepted'] = isAccepted
            inspection['taco_fail'] = taco_fail
            inspection['taco_pass'] = taco_pass
            inspection['timestamp'] = timestamp
            resp = inspection_attributes.update({'_id' : inspection['_id']}, {'$set' : inspection})
            cc = CacheHelper()
            rescan_status = False
            workstation_id = pr['workstation_id']
            rescan_status_key = RedisKeyBuilderServer(workstation_id).get_key(0, 'rescan-required')
            if rescan_status_key:
                cc.set_json({rescan_status_key : rescan_status}) 
                return resp
            else:
                return {}
        else:
            return {}
    else:
        return {}


def plan_production_counter_modify_util(data):
    mp = MongoHelper().getCollection('inspection_data')
    process_id = data.get('process_id')
    current_production_count = data.get('current_production_count')
    pr = mp.find_one({"_id" : ObjectId(process_id)})
    if pr:
        pr['current_production_count'] = current_production_count
        mp.update({'_id' : pr['_id']}, {'$set' : pr})
        return mp.find_one({"_id" : ObjectId(process_id)})
    else:
        return {}


def generate_QRcode_util(inspection_id):
    #mp = MongoHelper().getCollection('inspection_data')
    #fps = mp.find().sort([( '$natural', -1 )] )
    #fp = None
    #if fps.count() > 0:
    #    fp = fps[0]
    #process_id = str(fp['_id'])
    #inspection = MongoHelper().getCollection(str(process_id))
    #inspection_attributes = inspection.find_one({"_id" : ObjectId(inspection_id)})
    #qr_code = inspection_attributes['qr_string']
    #return qr_code
    inspection_collection = MongoHelper().getCollection(inspection_id)
    objs = inspection_collection.find().sort([( '$natural', -1 )] )
    if objs.count() > 0:
        obj = objs[0]
        return obj['qr_string']
    else:
        return ""


def redis_camera(key):
    rch = CacheHelper()
    while True:
        frame1 = rch.get_json(key)
        print("KEY :      : :: :  : : : : :" , key)
        ret, jpeg = cv2.imencode('.jpg', frame1)
        frame =  jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def get_inspection_qc_list(process_id):
    qc_inspection = []
    mp = MongoHelper().getCollection('inspection_data')
    pr = mp.find_one({'_id' : ObjectId(process_id)})
    if pr:
        if 'manual_inspection_result' in pr:
            qc_inpection = pr['manual_inspection_result']
            return qc_inpection
        else:
            return qc_inspection
    else:
        return ['process id not found.']