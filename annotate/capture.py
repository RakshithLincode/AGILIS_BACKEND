from common.utils import CacheHelper, MongoHelper, singleton
import cv2
import threading
import os
from livis import settings
import uuid


@singleton
class Camera:
    def __init__(self):
        #self.index = index
        self.cap = None
        self.curr_frame = None
        self.index = "/root/sample_data_livis_freedom/1.mp4"
    
    def init_camera(self):
        self.cap = cv2.VideoCapture(self.index)

    def get_frame(self):
        return self.curr_frame

    def service(self):
        while True:
            self.curr_frame = self.cap.read()[1]
            CacheHelper().set_json({'original_frame' :  self.curr_frame})

    def start_service(self):
        self.init_camera() 
        p = threading.Thread(target=self.service)
        p.start()

    def stop_service(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def save_frame(self, part_id):
        if self.curr_frame:
            mp = MongoHelper().getCollection(str(part_id))
            fname = str(uuid.uuid4()) + '.jpg'
            save_file_path = os.path.join(settings.TRAIN_DATA_STATIC, fname)
            cv2.imwrite(save_file_path, self.curr_frame)
            collection_obj = {
                    'file_path' : os.path.join(settings.TRAIN_DATA_STATIC,fname),
                    'file_url' : "http://0.0.0.0:3307/" + fname,
                    'state' : 'untagged',
                    'regions' : [],
                    'regions_history' : [],
                    'classifier_label' : "",
                    'classifier_label_history' : [],
                    'annotator' : ''
                    }
            inserted_id = mp.insert(collection_obj)
            return inserted_id


def start_camera():
    #try:
    c = Camera().start_service()
    return True
    #except:
    #    return False 


def stop_camera():
    try:
        c = Camera().stop_service()
        return True
    except:
        return False

def get_capture_feed_url():
    url = "http://0:0:0:0:8000/livis/v1/toyoda/stream1/original_frame/"
    return url


def capture_part_image(part_id):
    try:
        c = Camera().save_frame(part_id)
        if c:
            return {"message" : "Success!" , "inserted_frame" : c}
        else:
            return {"message" : "Failed!" }
    except:
        return {"message" : "Failed!" }







