import pymongo
from bson import ObjectId
import datetime

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["LIVIS"]
COLL_NAME = "INSPECTION_"+datetime.datetime.now().strftime("%m_%y")
mycol = mydb[COLL_NAME]


for x in mycol.find({"is_compleated":False}):
    try:
        result = mycol.delete_one({'_id': ObjectId(str(x['_id']))})
    except Exception as e:
        print(e)
