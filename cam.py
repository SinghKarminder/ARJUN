'''
@abstract: frames individually, mean count, csv .... 

@dscp:  this code creates csv every minute with mean_count of every seconds
        also saves images in which movement is detected

@algo: basic bg subtractor, no GLCM, no RGB plane norm

@author: vinay
'''

# use this to display bounding boxes
FRAME_DEBUG = False

# use this to show on console when files are created
LOG_DEBUG = False

#---------------------

import cv2
import numpy as np
from datetime import datetime
import json
import logging as log
import csv

log.basicConfig(filename='/var/tmp/cam.log', filemode='w', level=log.INFO, format='[%(asctime)s]- %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
log.info("Cam script started..")
with open(f"/etc/entomologist/ento.conf",'r') as file:
    data=json.load(file)

DEVICE_SERIAL_ID = data["device"]["SERIAL_ID"]
BUFFER_IMAGES_PATH = data["device"]["STORAGE_PATH"]
BUFFER_COUNT_PATH = data["device"]["COUNT_STORAGE_PATH"]

class MotionRecorder(object):
    
    VID_RESO = (1280,720)
    #VID_RESO = (640, 480)
    fps = 60

    # video capture : from device
    cap = cv2.VideoCapture(f"v4l2src device=/dev/video2 ! video/x-raw, width={VID_RESO[0]}, height={VID_RESO[1]}, framerate={fps}/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    #cap = cv2.VideoCapture("videotestsrc ! video/x-raw, format=I420, width=640, height=480 ! vpuenc_h264 ! appsink",cv2.CAP_GSTREAMER)

    # the background Subractors
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    #subtractor = cv2.createBackgroundSubtractorKNN()

    # FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')     # for windows
    
    CONTOUR_AREA_LIMIT = 10
    
    img_mean_persec_list = []    
    img_count_sum = 0
    img_count = 0
    # for storing frames as collection of 1 sec
    last_minute = None
    last_second = None

    def _init_(self):
        pass

    def process_img(self, frame):
        store = frame
        
        #------------------         
        # bg sub & blurring - Noise removal       
        imgOut = MotionRecorder.subtractor.apply(frame)
        imgOut = cv2.medianBlur(imgOut, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        imgOut1 = cv2.morphologyEx(imgOut, cv2.MORPH_OPEN, kernel,iterations = 2)
        imgOut1[imgOut1 == 127] = 0 # 0-> black
        
        imgOut2 = cv2.morphologyEx(imgOut1, cv2.MORPH_CLOSE, kernel,iterations = 2)
        # remove light variations(gray 127) - consider only Major Movement (white patches 255)    
        imgOut2[imgOut2 == 127] = 0 # 0-> black

        imgOut3 = cv2.morphologyEx(imgOut, cv2.MORPH_CLOSE, kernel,iterations = 2)
        # remove light variations(gray 127) - consider only Major Movement (white patches 255)    
        imgOut3[imgOut3 == 127] = 0 # 0-> black

        #------------------------w
        #draw contours
        contours, _ = cv2.findContours(imgOut2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detectionsRed = []
        for cnt in contours:
            area = cv2.contourArea(cnt)            
            x, y, w, h = cv2.boundingRect(cnt)
            x,y = x-5,y-5
            w,h = w+10,h+10

            if self.CONTOUR_AREA_LIMIT < area:
                detectionsRed.append([x, y, w, h])
        
        contours, _ = cv2.findContours(imgOut3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        numberofObjects = 0
        
        detections = []

        for cnt in contours:            
            area = cv2.contourArea(cnt)            
            x, y, w, h = cv2.boundingRect(cnt)
            x,y = x-5,y-5
            w,h = w+10,h+10

            if self.CONTOUR_AREA_LIMIT < area:
                # remove red countous
                enclosedRed = []
                thisEnclosedCount = 0
                for cntR in detectionsRed:
                    _x, _y, = cntR[:2]

                    if (x < _x < x+w) and (y <_y <y+h):
                        thisEnclosedCount+=1
                        enclosedRed.append(cntR)

                if thisEnclosedCount < 2:
                    detections.append([x, y, w, h])
                else:
                    detections.extend(enclosedRed)

        #------------------
        # display cntrs
        if FRAME_DEBUG:
            for cntr in detections:
                x,y,w,h = cntr
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 1)
                cv2.drawContours(store, [cnt], -1, (255,0,0),2)
                numberofObjects = numberofObjects + 1
                cv2.putText(store,str(numberofObjects), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_8)              

            cv2.putText(store,"Number of objects: " + str(len(detections)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_8)

        hasMovement = len(detections) > 0

        return hasMovement, frame, detections


    def start_storing_img(self, img):
        
        hasMovement, img2, bbox = self.process_img(img)
        if FRAME_DEBUG:
            img = img2
            #hasMovement = True

        # get current time
        now = datetime.now()

        # if change in minutes
        if self.last_minute == None:
            self.last_minute = now.minute
        elif self.last_minute != now.minute:
            # if sec changes, save last second count data            
            self.save_csv(now)
            #reset all
            self.last_minute = now.minute
            self.img_mean_persec_list = []
            self.img_count_sum = 0
            self.img_count = 0

        
        if hasMovement:

            # save image file
            self.temp_image_name = f'{now.strftime("%d-%m-%Y_%H-%M-%S-%f")}_{DEVICE_SERIAL_ID}.jpg'
            #self.save_recording(img)
            cv2.imwrite(BUFFER_IMAGES_PATH + self.temp_image_name, img)
            if LOG_DEBUG: print('Saved Image: ', self.temp_image_name, len(bbox))

            # assign count data                        
            self.img_count_sum += len(bbox)
            self.img_count += 1

        # if a change in second then append mean
        # save mean every second
        if self.last_second == None:
            self.last_second = now.second
        elif self.last_second != now.second:
            # if sec changes, save last second count data            
            
            if self.img_count != 0:
                mean_count_persec = self.img_count_sum // self.img_count
            else:
                mean_count_persec = 0

            self.img_mean_persec_list.append((DEVICE_SERIAL_ID, now.strftime("%d-%m-%Y_%H-%M-%S"), mean_count_persec))
            
            #reset all
            self.last_second = now.second                
            self.img_count_sum = 0
            self.img_count = 0
            
    
    def save_csv(self, timeNow):
        # save as count_timeFrame_deviceID_countMeanInt.csv
        # save as count_DD-MM-YYYY_hh-mm_DOxxx_XXXX.csv

        csvName = f'count_{timeNow.strftime("%d-%m-%Y_%H")}-{self.last_minute}_{DEVICE_SERIAL_ID}.csv'
        with open(BUFFER_COUNT_PATH + csvName, 'w',  newline='') as csvFile:
            csvwriter = csv.writer(csvFile)
            # header
            csvwriter.writerow(["device_id","time_frame","insect_count"])        
            # data 
            csvwriter.writerows(self.img_mean_persec_list)
    
        log.info("Video bbox count CSV crealog.info("")ted and saved -> "+csvName)
        if LOG_DEBUG: print('CSV saved', csvName)
            
            


    def save_recording(self, image):
        pass
        #cv2.imwrite(BUFFER_IMAGES_PATH+self.temp_image_name, image)

            
    def start(self):
        log.info("Cam started functioning")

        #fCount = 0

        while True :
            available, frame = self.cap.read()

            if available:                
                self.start_storing_img(frame)

                # check exit
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    break
            else:
                if FRAME_DEBUG:
                    print("...Device Unavailable");

    def end(self):        
        self.save_recording()
        self.cap.release()
        cv2.destroyAllWindows()


# main
MR = MotionRecorder()
log.info("Object created")
MR.start()
MR.end()
log.info("Script ended")

# to send this file over ssh to device 
# scp {source_path} root@192.168.8.1:/usr/sbin/cam/cam.py
# scp /home/tif-awadh/Desktop/local_see3cam_test/device_code/microseconds/cam.py root@192.168.8.1:/usr/sbin/cam/cam.py
