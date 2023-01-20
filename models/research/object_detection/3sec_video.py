import numpy as np
import cv2
import time

# Define the duration (in seconds) of the video capture here
capture_duration = 3

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('3s.avi',fourcc, 20.0, (640,480))

start_time = time.time()
i= 0
while( int(time.time() - start_time) < capture_duration ):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)
        cv2.imwrite('/home/lincode/Desktop/GORAD/faster_RCNN/cataler/pluggedcells/depth/models/research/object_detection/t_img/img_'+str(i)+'.jpg',frame)
        i = i+1
        cv2.imshow('frame',frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


# # Importing all necessary libraries 
# import cv2 
# import os 

# # Read the video from specified path 
# cam = cv2.VideoCapture("/home/lincode/Desktop/GORAD/faster_RCNN/cataler/pluggedcells/depth/models/research/object_detection/3s.avi") 

# try: 
    
#     # creating a folder named data 
#     if not os.path.exists('data'): 
#         os.makedirs('data') 

# # if not created then raise error 
# except OSError: 
#     print ('Error: Creating directory of data') 

# # frame 
# currentframe = 0

# while(True): 
    
#     # reading from frame 
#     ret,frame = cam.read() 

#     if ret: 
#         # if video is still left continue creating images 
#         name = './data/frame' + str(currentframe) + '.jpg'
#         print ('Creating...' + name) 

#         # writing the extracted images 
#         cv2.imwrite(name, frame) 

#         # increasing counter so that it will 
#         # show how many frames are created 
#         currentframe += 1
#     else: 
#         break

# # Release all space and windows once done 
# cam.release() 
# cv2.destroyAllWindows() 
