import camera_module
import cv2
# def config_file1():
#     baumer_ip = []
#     baumer_ip.append("192.168.1.4")
#     baumer_ip.append("192.168.1.3")
#     return baumer_ip

# baumer_ip = config_file1()
# # this config is for baumer 
# cam_1 = camera_module.camera('baumer',baumer_ip[0])
# cam_2 = camera_module.camera('baumer',baumer_ip[1])

# this config is for Lucid 
cam_1 = camera_module.Lucid('221501115','top')
# cam_2 = camera_module.Lucid('221501115','bottom')

while True:
    try:
        camera_1 = cam_1.fetch_cameras()
        if camera_1 is None:
            print("camera_1 is not connected to device") 
        # camera_2 = cam_2.fetch_cameras()
        # if camera_2 is None:
        #     print("camera_2 is not connected to device")    
        # frame_2 = imutils.rotate(frame_2,180)
        camera_1 = cv2.resize(camera_1,(640,480))
        cv2.imwrite('framme.png',camera_1)

        cv2.imshow('framme',camera_1)
        # cv2.destroyAllWindows() 

        # rch.set_json({'frame1':camera_1})
        # rch.set_json({'frame2':camera_2})
        # print("elapsed to write"+ str(d-s))
        
    except Exception as e:
        print(e)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
cv2.destroyAllWindows() 



