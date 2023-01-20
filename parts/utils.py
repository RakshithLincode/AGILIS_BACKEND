from common.utils import MongoHelper
from livis.settings import *
from bson import ObjectId
from plan.utils import get_todays_planned_production_util
from common.utils import GetLabelData
from datetime import date
from datetime import datetime


#######################################################################PART CRUDS#######################################################
def add_part_details_task(data):
    """
    {
	"short_number": "sht11",
	"model_number": "md11",
	"part_number": "pt11",
	"planned_production": "100",
	"part_description": "fjjff",
	"edit_part_data": true
    }
    """
    now = datetime.now()

# dd/mm/YY
    try:
        short_number = data.get('short_number',None)
        model_number = data.get('model_number',None)
        planned_production = data.get('planned_production',None)
        part_number = data.get('part_number',None)
        part_description = data.get('part_description',None)
        edit_part_data = data.get('edit_part_data',None)
        kanban = data.get('kanban',None)
        isdeleted = False
        current_date_time = now.strftime("%d/%m/%Y %H:%M")

        mp = MongoHelper().getCollection(PARTS_COLLECTION)

        collection_obj = {
           'short_number' : short_number,
           'model_number' : model_number,
           'planned_production' : planned_production,
           'part_number' : part_number,
           'part_description' : part_description,
           'edit_part_data' : edit_part_data,
           'isdeleted' : isdeleted,
           'kanban' : kanban,
           'created_on': current_date_time,
           'modified_on': current_date_time
        }
        _id = mp.insert(collection_obj)
        return _id
    except Exception as e:
        return "Could not add part: "+str(e)


def delete_part_task(part_id):
    _id = part_id
    mp = MongoHelper().getCollection(PARTS_COLLECTION)
    p = mp.find_one({'_id' : ObjectId(_id)})
    if p:
        isdeleted = p.get('isdeleted')
        if not isdeleted:
            p['isdeleted'] = True
        mp.update({'_id' : p['_id']}, {'$set' :  p})
        return _id
    else:
        return "Part not found."


def update_part_task(data):
    """
    {
        "_id": "242798143hdw7q33913413we2"
	    "short_number": "sht11",
	    "model_number": "md11",
	    "part_number": "pt11",
	    "planned_production": "100",
	    "part_description": "fjjff",
	    "edit_part_data": true
    }
    """
    _id = data.get('_id')
    now = datetime.now()

    if _id:
        mp = MongoHelper().getCollection(PARTS_COLLECTION)
        pc = mp.find_one({'_id' : ObjectId(_id)})
        if pc:
            short_number = data.get('short_number',None)
            model_number = data.get('model_number',None)
            part_number = data.get('part_number',None)
            planned_production = data.get('planned_production',None)
            part_description = data.get('part_description',None)
            edit_part_data = data.get('edit_part_data',None)
            current_date_time = now.strftime("%d/%m/%Y %H:%M")
            if short_number:
                pc['short_number'] = short_number
            if model_number:
                pc['model_number'] = model_number
            if part_number:
                pc['part_number'] = part_number
            if planned_production:
                pc['planned_production'] = planned_production
            if part_description:
                pc['part_description'] = part_description
            if edit_part_data:
                pc['edit_part_data'] = edit_part_data
            if kanban:
                pc['kanban'] = kanban
            pc['modified_on'] = current_date_time
            mp.update({'_id' : pc['_id']}, {'$set' :  pc})
        else:
            return {"message" : "Part not found"}
        return _id
    else:
        return {"message" : "Please enter the part ID."}


def get_part_details_task(part_id):
    _id = ObjectId(part_id)
    mp = MongoHelper().getCollection(PARTS_COLLECTION)
    p = mp.find_one({'_id' : _id})
    if p:
        return p
    else:
        return {}


def get_parts_task(skip=0, limit=100):
    mp = MongoHelper().getCollection(JIG_COLLECTION)
    
    parts = [p for p in mp.find({"$and" : [{"is_deleted": False}, { "is_deleted" : {"$exists" : True}}]}).sort( "$natural", -1 )]
    #print(parts)

    #parts = [p for p in mp.find({"$and" : [{"isdeleted": False}, { "isdeleted" : {"$exists" : True}}]}).skip(skip).limit(limit)]

    for i in parts:
        data = {}
        part_obj_id = i["_id"]
        mp = MongoHelper().getCollection(str(part_obj_id) + '_experiment')
        i["experiments"] = [i for i in mp.find()]
        info = GetLabelData(part_obj_id).get_metrics()
        i["label_info"] = info
    if parts:
        return parts
    else:
        return []


def get_partInfo(short_number):      #part_info based on short_number
    mp = MongoHelper().getCollection(PARTS_COLLECTION)
    pr = [i for i in mp.find({"short_number" : short_number})]
    resp = {}
    if len(pr) > 0:
        resp = pr[-1]
        obj = get_todays_planned_production_util(short_number)
        planned_production = obj['planned_production_count']
        resp.update({'planned_production':planned_production})
        return resp
    else:
        return {}


def get_short_numbers_list_util(skip=0, limit=100):
    mp = MongoHelper().getCollection(PARTS_COLLECTION)
    short_numbers = [p for p in mp.find({"$and" : [{"isdeleted": False}, { "isdeleted" : {"$exists" : True}}]},{"short_number":1, "_id":0}).skip(skip).limit(limit)]
    if short_numbers:
        return short_numbers
    else:
        return []

        
