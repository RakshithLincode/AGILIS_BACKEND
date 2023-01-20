from IO_Module import *
import redis
import pickle

def singleton(cls):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

@singleton
class CacheHelper():
    def __init__(self):
        self.redis_cache = redis.StrictRedis(host="127.0.0.1", port="6379", db=0, socket_timeout=1)
        #self.redis_cache = redis.StrictRedis(host=s.REDIS_CLIENT_HOST, port=s.REDIS_CLIENT_PORT, db=0, socket_timeout=1)
        #s.REDIS_CLIENT_HOST
        print("REDIS CACHE UP!")

    def get_redis_pipeline(self):
        return self.redis_cache.pipeline()
    
    def set_json(self, dict_obj):
        try:
            k, v = list(dict_obj.items())[0]
            v = pickle.dumps(v)
            return self.redis_cache.set(k, v)
        except redis.ConnectionError:
            return None

    def get_json(self, key):
        try:
            temp = self.redis_cache.get(key)
            #print(temp)\
            if temp:
                temp= pickle.loads(temp)
            return temp
        except redis.ConnectionError:
            return None
        return None

    def execute_pipe_commands(self, commands):
        #TBD to increase efficiency can chain commands for getting cache in one go
        return None
        
        
plc_address = { 'start_process':2050,
                'accepted':2048,
                'rejected':2049}

plc = plc_io()

while 1:
    start_process = CacheHelper().get_json('plc_insp_started')
    if start_process:
        print("process started !!!!!!!!!")
        plc.write(plc_address['start_process'],1)
        CacheHelper().set_json({'plc_insp_started':None})
        
    part_status = CacheHelper().get_json('plc_insp_status')
    if part_status is True:
        print("got accepted !!!!!!")
        plc.write(plc_address['accepted'],1)
        CacheHelper().set_json({'plc_insp_status':None})
        
    elif part_status is False :
        print("got Rejected !!!!!!")
        plc.write(plc_address['rejected'],1)
        CacheHelper().set_json({'plc_insp_status':None})
        
    elif part_status is None :
        continue
