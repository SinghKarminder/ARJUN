# use this to show bounding box on image
DISPLAY_CONTOURS = False

import cv2 as cv
import numpy as np
from datetime import datetime
import json
import logging as log
from skimage.feature import graycomatrix, graycoprops

log.basicConfig(filename='/var/tmp/cam.log', filemode='w', level=log.INFO, format='[%(asctime)s]- %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
log.info("Cam script started..")
with open(f"/etc/entomologist/ento.conf",'r') as file:
    data=json.load(file)

DEVICE_SERIAL_ID = data["device"]["SERIAL_ID"]
BUFFER_IMAGES_PATH = data["device"]["STORAGE_PATH"]

class MotionRecorder(object):
    
    #VID_RESO = (1280,720)
    VID_RESO = (640, 480)

    # video capture : from device
    #cap = cv.VideoCapture("v4l2src device=/dev/video2 ! video/x-raw, width=1280, height=720, framerate=60/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv.CAP_GSTREAMER)
    cap = cv.VideoCapture(f"v4l2src device=/dev/video2 ! video/x-raw, width={VID_RESO[0]}, height={VID_RESO[1]}, framerate=60/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv.CAP_GSTREAMER)
    #cap = cv.VideoCapture("videotestsrc ! video/x-raw, format=I420, width=640, height=480 ! vpuenc_h264 ! appsink",cv.CAP_GSTREAMER)

    # the background Subractors
    subtractor = cv.createBackgroundSubtractorMOG2()
    #subtractor = cv.createBackgroundSubtractorKNN()

    # FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org.
    fourcc = cv.VideoWriter_fourcc(*'DIVX')     # for windows
    fps = 60

    IMAGE_COUNTER_LIMIT = 10
    img_counter = 0

    CONTOUR_AREA_LIMIT = 10
    
    temp_img_for_video = []
    temp_img_bbox_for_video = {}
    
    # red blue colour intensity difference threshold
    RB_threshold = 30

    def _init_(self):
        pass
    
    def process_img(self, frame):        
        #------------------         
        # bg sub & blurring - Noise removal       
        imgOut = MotionRecorder.subtractor.apply(frame)
        imgOut = cv.medianBlur(imgOut, 3)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

        # remove shadow


        imgOut1 = cv.morphologyEx(imgOut, cv.MORPH_OPEN, kernel,iterations = 2)
        imgOut1[imgOut1 == 127] = 0 # 0-> black
        
        imgOut2 = cv.morphologyEx(imgOut1, cv.MORPH_CLOSE, kernel,iterations = 2)
        # remove light variations(gray 127) - consider only Major Movement (white patches 255)    
        imgOut2[imgOut2 == 127] = 0 # 0-> black

        imgOut3 = cv.morphologyEx(imgOut, cv.MORPH_CLOSE, kernel,iterations = 2)
        # remove light variations(gray 127) - consider only Major Movement (white patches 255)    
        imgOut3[imgOut3 == 127] = 0 # 0-> black

        #------------------------w
        #draw contours
        contours, _ = cv.findContours(imgOut2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        detectionsRed = []
        for cnt in contours:
            #area = cv.contourArea(cnt)            
            x, y, w, h = cv.boundingRect(cnt)            

            objinconsider = frame[y:y+h,x:x+w]
            image = cv.cvtColor(objinconsider, cv.COLOR_BGR2GRAY)
            glcm = graycomatrix(image,distances=[3],angles=[0,np.pi/4,np.pi/2],levels=256,symmetric=True,normed=True)
            
            dis=graycoprops(glcm,'contrast')[0,0]
#                 cor=graycoprops(glcm,'correlation')[0,0]
            if dis > 800:
                x,y = x-5,y-5
                w,h = w+10,h+10
                detectionsRed.append([x, y, w, h])
        
        contours, _ = cv.findContours(imgOut3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        detections = []

        for cnt in contours:            
            #area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)

            
            objinconsider = frame[y:y+h,x:x+w]
            image = cv.cvtColor(objinconsider, cv.COLOR_BGR2GRAY)
            glcm = graycomatrix(image,distances=[3],angles=[0,np.pi/4,np.pi/2],levels=256,symmetric=True,normed=True)
            
            dis=graycoprops(glcm,'contrast')[0,0]

            if dis > 800:
                x,y = x-5,y-5
                w,h = w+10,h+10

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

                detectionsRed.append([x, y, w, h])
                      
        #------------------
        # display cntrs
        if DISPLAY_CONTOURS:
            for cntr in detections:
                x,y,w,h = cntr
                cv.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 1)
        
        # if has boxes => valid frame else skip
        hasMovement = len(detections) > 0

        return hasMovement, frame, detections


    def start_storing_img(self, img):
        '''@vinay
        draw bounding box
        + save bbox json file

        return None'''

        hasMovement, img, bbox = self.process_img(img)
        
        if hasMovement:            
            self.temp_img_for_video.append(img)
            self.temp_img_bbox_for_video[self.img_counter] = bbox
            self.img_counter += 1
            if self.img_counter > self.IMAGE_COUNTER_LIMIT:
                self.save_recording()
    
    def save_recording(self):
        if self.img_counter >= 1:   
            now = datetime.now()
            video_name = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}.avi'  
            out = cv.VideoWriter(BUFFER_IMAGES_PATH+video_name, self.fourcc, self.fps, (MotionRecorder.VID_RESO[0],MotionRecorder.VID_RESO[1]))
                        
            for image in self.temp_img_for_video : 
                out.write(image)

            # log data
            log.info("Video crealog.info("")ted and saved -> "+video_name)            
            print(video_name)

            # json file name
            json_fname = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}.json'            
            json_file = open(BUFFER_IMAGES_PATH+json_fname, 'w')

            # save json bbox
            json.dump(self.temp_img_bbox_for_video, json_file)
            json_file.close()

            # log data
            log.info("Video bbox JSON crealog.info("")ted and saved -> "+json_fname)
            print(json_fname)

            # reset all
            self.temp_img_for_video.clear()
            self.temp_img_bbox_for_video.clear()
            self.img_counter = 0


    def start(self):
        log.info("Cam started functioning")

        while True :
            available, frame = self.cap.read()

            if available:                
                self.start_storing_img(frame)
                
                # check exit
                if cv.waitKey(1) & 0xFF == ord('x'):
                    break
            else:
                if DISPLAY_CONTOURS:
                    print("...Device Unavailable");

    def end(self):        
        self.save_recording()
        self.cap.release()
        cv.destroyAllWindows()


# main
MR = MotionRecorder()
log.info("Object created")
MR.start()
MR.end()
log.info("Script ended")
