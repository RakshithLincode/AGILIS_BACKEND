import os
import shutil
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import glob

from annotate.salient_detector.data_loader import RescaleT
from annotate.salient_detector.data_loader import ToTensor
from annotate.salient_detector.data_loader import ToTensorLab
from annotate.salient_detector.data_loader import SalObjDataset

from annotate.salient_detector.model import U2NET

net = None


def throw_dest_path():

    image_dir = os.path.join(os.getcwd(), 'annotate', 'salient_detector', 'test_data', 'test_images')
    return image_dir

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+'a'+'.png')

def load_model():
    global net
    model_name='u2net'
    print("...loading U2NET to the memory")
    model_dir = os.path.join(os.getcwd(), 'annotate', 'salient_detector', 'saved_models', model_name, model_name + '.pth')
    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir, map_location="cpu"))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    return net

def load_data():

    model_name='u2net'

    image_dir = os.path.join(os.getcwd(), 'annotate', 'salient_detector', 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'annotate', 'salient_detector', 'test_data', model_name + '_results' + os.sep)

    shutil.rmtree(prediction_dir)
    os.makedirs(prediction_dir)

    img_name_list = glob.glob(image_dir + os.sep + '*')

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    return prediction_dir,img_name_list,test_salobj_dataloader

def run_inference(prediction_dir,img_name_list,test_salobj_dataloader):

    global net
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

def get_cords(prediction_dir,size):

    img = cv2.imread(prediction_dir+"a.png", 0)
    canny_img = cv2.Canny(img, 80, 150)
    contours, hierarchy = cv2.findContours(canny_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cords_list = []
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>size:
            x,y,w,h = cv2.boundingRect(contour)
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            cords_list.append([x1,y1,x2,y2])
            
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < size or rect[3] < size: continue
        print(cv2.contourArea(c))
        x,y,w,h = rect

    return cords_list



def main():


    #load model
    net = load_model()

    #load dataloader
    prediction_dir,img_name_list,test_salobj_dataloader = load_data()
    
    #inference
    run_inference(net,prediction_dir,img_name_list,test_salobj_dataloader)

    #process contours
    cords_list = get_cords(prediction_dir,min_height,min_width,max_height,max_width)
    print(cords_list)

#main()
