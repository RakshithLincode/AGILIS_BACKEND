from flask import Flask,request,jsonify
import json
from flask_5001_utils import *
from iteration_utilities import unique_everseen
import time

img_path = r"D:\24JAN\VCXG-201C.R\DATA\image0000035.jpg"
oem_number = 'ON-2213'

app = Flask(__name__) 

@app.route('/predict',methods = ['POST', 'GET'])
def pred(): 
    print("inside predict")
    a = time.time()
    results_label = get_ocr(img_path,oem_number)
    actual_value = get_kanban(oem_number)
    predicted_value = list(unique_everseen(results_label))
    position  = check_kanban(actual_value,predicted_value)
    print(position,'kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
    print("totaltime",str(time.time()-a))
    return json.dumps({'prediction':predicted_value,'position':position})
  
if __name__ == '__main__': 
    app.run(host='127.0.0.1',port=5001,debug=True) 
