from common.utils import *
from django.utils import timezone
from bson import ObjectId
from toyoda.utils import generate_QRcode_util
from toyoda.utils import get_inspection_qc_list
from copy import deepcopy
import sqlite3
from xlsxwriter import Workbook
import datetime
import pymongo
from bson import ObjectId
import datetime
from datetime import date
import calendar
from csv import DictWriter
from datetime import datetime, timedelta
from dateutil.relativedelta import *
from datetime import date





#nss
def detail_report_util(data):
    try:
        from_date = data['from_date']
    except:
        from_date = None
    try:
        to_date = data['to_date']
    except:
        to_date = None
    try:
        jig_type = data['feature_type']
    except:
        jig_type = None
    try:
        operator_name = data['operator_name']
    except:
        operator_name = None    
    try:
        status_end = data['status_type']
    except:
        status_end = None  
        
    try:
        on_number = data['on_number']
    except:
        on_number = None  
    try:
        serial_number = data['serial_number']
    except:
        serial_number = None  		
            


    query_1 = []
    query_2 = []
    resp = {}
    resp_list = []


    today = date.today()
    day = today.strftime("%d")
    month = today.strftime("%m")
    year = today.strftime("20%y")
    start = (int(year), int(month), int(day))
    end = today + relativedelta(months=-3)
    day = end.strftime("%d")
    month = end.strftime("%m")
    year = end.strftime("20%y")
    end = (int(year), int(month), int(day))

    # start = datetime.date(2021, 12, 23)
    # print(start)
    # end = datetime.date(2021, 12, 23)
    # print(end)
    
    if (from_date is not None) :
        query_1.append({'createdAt': {"$gte":from_date}})#,"$lte":to_date
    elif (from_date is None) :
        query_1.append({'createdAt': {"$gte":str(start)}})#,"$lte":to_date
    if (to_date is not None) :
        query_1.append({'completedAt': {"$lte":to_date}})
    elif (to_date is None) :
        query_1.append({'completedAt': {"$lte":str(end)}})
    if jig_type is not None :
        query_1.append({'jig_details.jig_type': jig_type})
    if on_number is not None :
        query_1.append({'jig_details.oem_number': on_number})
    if operator_name is not None :
        query_1.append({'user.user_id': operator_name})
    if status_end is not None:
        query_1.append({'status_end':status_end})
    if serial_number is not None:
        query_1.append({'serial_no':serial_number})		
    print(operator_name)
    print(serial_number,'ssssssssssssssssssssssssssssssssssss')
    print(query_1)
    

    
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["LIVIS"]
    
    all_lst = mydb.collection_names()
    
    insp_lst = []
    
    for a_lst in all_lst:
        if 'INSPECTION' in a_lst:
             insp_lst.append(a_lst)
    
    for i_lst in insp_lst:
    
        #COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
        process_collection = MongoHelper().getCollection(i_lst)
        
        
        if bool(query_1):
            pr_ids = [i['_id'] for i in process_collection.find({"$and":query_1})]
        else:
            pr_ids = [i['_id'] for i in process_collection.find()]
        

        for ind , pr_id in enumerate(pr_ids):
            res = process_collection.find({"_id":pr_id})
            #print(pr_id)
            #print(res)
            for r in res:
                if 'approved_by' in list(r.keys()):
                
                    name = get_name_byid(r["approved_by"])
                
                    resp_list.append({"id":ind,
                                "on_number": r["jig_details"]["oem_number"],
                                'operator_name': r['user']['name'] ,
                                'jig_type':r["jig_details"]["jig_type"],
                                "serial_number":r["serial_no"],
                                'scanned_at':r["createdAt"],
                                # "completed_at":r["completedAt"],
                                "status":r["status_end"],
                                "num_retry":r["num_retry"],
                                "approved_by":name})
                else:
                    resp_list.append({"id":ind,
                        "on_number": r["jig_details"]["oem_number"],
                        'operator_name': r['user']['name'] ,
                        'jig_type':r["jig_details"]["jig_type"],
                        "serial_number":r["serial_no"],
                        'scanned_at':r["createdAt"],
                        # "completed_at":r["completedAt"],
                        "num_retry":r["num_retry"],
                        "status":r["status_end"],
                        "approved_by":None})
                   
                   
                   
    return resp_list


def get_name_byid(id):

    #print(id)

    command = "SELECT * FROM accounts_user WHERE user_id=" + '\"' + id + '\"'
    #print(command)

    conn = sqlite3.connect('D:/SE_PROJECT/livis-be-se-agilis_be/livis-be-se-agilis_be/db.sqlite3')
    cursor = conn.cursor()
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    cursor.execute(command)
    lis = cursor.fetchone()
    
    #print(lis)
    #admin_name = "N.A"  
    
    
    if lis is not None and len(lis)>0:
    
        return str(lis[4])
    else:
        return "N.A"
    

def operator_list():
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    cursor.execute("SELECT * FROM accounts_user")
    lis = cursor.fetchall()
#     print(lis)
    list_dict = []
#     print(len(lis))
    for l in lis:
#         print(list(l))
        if "operator" == l[14]:
            a = {"user_id":l[3],
            "role":"operator",
            "operator_name": str(l[4] +" "+ l[5]),#l[4],
#             "operator_id":l[6]
                }
            list_dict.append(a)
    print(list_dict,'ddddddddddddddddddddddddddddddddddddddddddddddddddddd')        
    return list_dict


def write_excel(list_dict, file_name):
    if not list_dict:
        list_dict = [{"id":'',"on_number":"","operator_name":"","serial_number":"","scanned_at":"","num_retry":"","status":"","approved_by":""}]
    else:
        pass
    ordered_list=list(list_dict[0].keys())
    pth = "D:/SE_PROJECT/livis-be-se-agilis_be/livis-be-se-agilis_be/datadrive/" + file_name + ".xlsx"
    wb=Workbook(pth)
    ws=wb.add_worksheet("New Sheet") #or leave it blank, default name is "Sheet 1"
    first_row=0
    for header in ordered_list:
        col=ordered_list.index(header) 
        ws.write(first_row,col,header) 
    row=1
    for dic in list_dict:
#         print(dic)
#         print(dic.items())
        for _key,_value in dic.items():
            col=ordered_list.index(_key)
            ws.write(row,col,_value)
        row+=1 #enter the next row
    wb.close()
    return file_name + '.xlsx'

def export_file(list_dict,file_name):
    fn = write_excel(list_dict, file_name)
    return "http://localhost:3306/"+fn

def detail_report_export_util(data,path ="D:/SE_PROJECT/livis-be-se-agilis_be/livis-be-se-agilis_be/datadrive/"):
    list_dict = detail_report_util(data)
    import bson
    bsf = str(bson.ObjectId())
    # file_name = os.path.join(path,"report_details")
    file_name = "report_details" + bsf
    fn = export_file(list_dict,file_name)
    return fn

def get_last_defect_list_util(inspection_id,skip=0, limit=20):
    inspection_collection = MongoHelper().getCollection(inspection_id)
    objs = inspection_collection.find().sort([( '$natural', -1 )] )
    if objs.count() > 0:
        obj = objs[0]
        print("-----obj: ", obj)
        base64_image = generate_QRcode_util(inspection_id)
        obj.update({'qr_string' : base64_image})
        return obj
    else:
        return {}

def get_metrics_util(inspection_id):
    mp = MongoHelper().getCollection('inspection_data')
    pr = mp.find_one({"_id" : ObjectId(inspection_id)})
    print(pr)
    if pr:
        wid = pr['workstation_id']
        camera_id = 9
        workstation = MongoHelper().getCollection('workstations').find_one({'_id' : ObjectId(wid)})
        camera_config = [i['camera_id'] for i in workstation['cameras'] if i['camera_name'] != 'kanban']
        if len(camera_config) > 0:
            camera_id = camera_config[0]
        rescan_status_key = RedisKeyBuilderServer(wid).get_key(camera_id, 'rescan-required')
        if rescan_status_key:
            cc = CacheHelper()
            rescan_status = cc.get_json(rescan_status_key)
            print("REsCAN STATUS KEY : : :: : : : : : ", rescan_status_key, " ; ; ; ; " , rescan_status)
            inspection = MongoHelper().getCollection(inspection_id)
            total = inspection.count()
            total_accepted = inspection.find({'isAccepted' : True}).count()
            total_rejected = total - total_accepted
            qc_inspection = get_inspection_qc_list(inspection_id)
            resp = {
                "accepted" : total_accepted,
                "rejected" : total_rejected,
                "total" : total,
                "rescan_status" : rescan_status,
                "qc_inspection" : qc_inspection
            }
            return resp
        else:
            return {}
    else:
        return {}


def get_accepted_rejected_parts_list_util(start_date=None, end_date=None, status=None):
    mp = MongoHelper().getCollection('inspection_data')
    pr = [i['_id'] for i in mp.find()]
    all_accepted_parts = []
    all_rejected_parts = []
    all_parts = []    
    for id in pr:
        cp = MongoHelper().getCollection(str(id))
        accepted_parts = [i for i in cp.find({"isAccepted" : True})]
        rejected_parts = [i for i in cp.find( {"$or" : [
            {"isAccepted": False},
            { "isAccepted" : {"$exists" : False}}
            ]})]
        all_parts.extend([i for i in cp.find()])
        all_accepted_parts.extend(accepted_parts)
        all_rejected_parts.extend(rejected_parts)
    if status:
        if status:
            resp = {"data"  : all_accepted_parts}
        else:
            resp = {"data" : all_rejected_parts}
    else:
        resp = {"data" : all_parts}
    return resp





def get_summary_end_process_util(inspection_id):
    resp = {}
    mp = MongoHelper().getCollection('inspection_data')
    process_attributes = mp.find_one({'_id' : ObjectId(inspection_id)})
    if process_attributes:
        user_image_url = ""
        collection_obj = {
           'part_number' : process_attributes['part_number'],
           'model_number' : process_attributes['model_number'],
           'short_number' : process_attributes['short_number'],
           'operator_name' : process_attributes['user']['name'],
           'createdAt' : process_attributes['createdAt'],
           'completedAt' : process_attributes['completedAt'],
           'duration' : process_attributes['duration'],
           'operator_role' : process_attributes['user']['role'],
           'total_parts' : process_attributes['total_parts'],
           'total_accepted_parts' : process_attributes['total_accepted_parts'],
           'total_rejected_parts' : process_attributes['total_rejected_parts'],
           'user_image_url' : user_image_url
        }
        resp = collection_obj
        return resp
    else:
        return "Data not found."


def defect_type_based_report_util(data):
    from_date = data.get('from_date')
    to_date = data.get('to_date')
    workstation_id = data.get('workstation_id',None)
    mp = MongoHelper().getCollection('inspection_data')
    query = []
    query.append({'timestamp': {"$gte":from_date,"$lte":to_date}})
    all_defect_list = []
    total_count = 0
    if workstation_id:
        pr = [i for i in mp.find({'workstation_id': workstation_id})]
    else :
        pr = [i for i in mp.find()]
    master_defects_list = get_master_defects()
    #print("----pr: ",pr)
    for defect in master_defects_list:
        #print("--------------------------Loop---------------------")
        #print("----defect: ",defect)
        for obj in pr:
            #print("----obj: ",obj)
            id1 = obj['_id']
            
            inspectionid_collection = MongoHelper().getCollection(str(id1)) 
            defect_counter =  inspectionid_collection.count(
            {
                "$and": [{'timestamp': {"$gte":from_date,"$lte":to_date}},{'defect_list': {"$in" : [defect]}}]
            })
            #print("----defect_counter: ",defect_counter)
            total_count+=defect_counter
        collection_obj = {
            'defect_type' : defect,
            'count' : total_count
            }    
        #print("----collection_obj: ",collection_obj)
        all_defect_list.append(collection_obj)
        defect_counter = 0
        total_count = 0    
    return all_defect_list


def get_master_defects():
    #mp = MongoHelper().getCollection(PARTS_COLLECTION)
    #pr = [i for i in mp.find()]
    #master_defect_list = []
    #for p in pr:
    #    if 'kanban' in p:
    #        if 'defect_list' in p['kanban']:
    #            master_defect_list.extend(p['kanban']['defect_list'])
    #return list(set(master_defect_list))
    master_defect_list = ['Shot_Shot_Presence']
    return master_defect_list


def get_master_features():
    #mp = MongoHelper().getCollection(PARTS_COLLECTION)
    #pr = [i for i in mp.find()]
    #master_feature_list = []
    #for p in pr:
    #    if 'kanban' in p:
    #        if 'feature_list' in p['kanban']:
    #            master_feature_list.extend(p['kanban']['feature_list'])
    #return list(set(master_feature_list))
    master_feature_list = ['PPWOSRSRH','PPWOSRSLH','Felt_Presence','Clip_Presence']
    return master_feature_list
