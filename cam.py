from PIL import Image
import cv2
import numpy as np
import threading
from datetime import datetime
import json
import logging as log

log.basicConfig(filename='/var/tmp/cam.log', filemode='w', level=log.INFO, format='[%(asctime)s]- %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
log.info("Cam script started..")
with open(f"/etc/entomologist/ento.conf",'r') as file:
    data=json.load(file)

DEVICE_SERIAL_ID = data["device"]["SERIAL_ID"]
BUFFER_IMAGES_PATH = data["device"]["STORAGE_PATH"]

class MotionRecorder(object):

    IMAGE_COUNTER_LIMIT = 100
    CONTOUR_AREA_LIMIT = 200
    FRAMES_TO_SKIP = 5
    

    hist_threshold = 500    # motion sensitivity => higher the value lesser the sensitivity
    #path = 0
    
    cap = cv2.VideoCapture("v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480, framerate=60/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    #cap = cv2.VideoCapture("videotestsrc ! video/x-raw, format=I420, width=640, height=480 ! vpuenc_h264 ! appsink",cv2.CAP_GSTREAMER)
    
    #subtractor = cv2.createBackgroundSubtractorMOG2()
    subtractor = cv2.createBackgroundSubtractorKNN()
    # FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')     # for windows
    fps = 60
    img_counter = 0
    skip_counter = 0
    temp_img_for_video = []
    temp_img_bbox_for_video = {}
    skip_first_few_frames = 0

    def _init_(self):
        pass

    def process_img_new(self, img):
        '''@vinay
        process image frames
        
        return: hasimg'''
        
        mask = self.subtractor.apply(img)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations = 4)

        # remove minute variations(gray 127) - consider only Major Movement (white patches 255)    
        mask[mask == 127] = 0 # 0-> black

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > self.CONTOUR_AREA_LIMIT:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        #cv2.imshow("Frame", img)
        #cv2.imshow("Mask", mask)           

        # store frame details in json object
        # frameIndex X1,X2,Y1,Y2
        # todo;;;;;       
        cv2.destroyAllWindows()

        # if has boxes => valid frame else skip
        hasMovement = len(detections) > 0

        return hasMovement, img, detections

    def start_storing_img_new(self, img):
        '''@vinay
        draw bounding box
        +can display on screen 
        
        . save bbox json file

        return None'''

        hasMovement, img, bbox = self.process_img_new(img)
        
        if hasMovement:            
            self.temp_img_for_video.append(img)
            self.temp_img_bbox_for_video[self.img_counter] = bbox
            self.img_counter += 1
            if self.img_counter > self.IMAGE_COUNTER_LIMIT:
                self.save_recording()

    def start_storing_img(self, img):

        # image processing
        blur = cv2.GaussianBlur(img, (19,19), 0)
        mask = self.subtractor.apply(blur)
        img_temp = np.ones(img.shape, dtype="uint8") * 255
        img_temp_and = cv2.bitwise_and(img_temp, img_temp, mask=mask)
        img_temp_and_bgr = cv2.cvtColor(img_temp_and, cv2.COLOR_BGR2GRAY)

        hist, bins = np.histogram(img_temp_and_bgr.ravel(), 256, [0,256])
        #print(hist[255])

        # validate and store frames
        if(self.skip_first_few_frames < 5) : 
            self.skip_first_few_frames += 1
        else : 
            if hist[255] > self.hist_threshold:
                self.skip_counter = 0
                self.img_counter += 1 
                self.temp_img_for_video.append(img)
                if self.img_counter > self.IMAGE_COUNTER_LIMIT:
                    self.save_recording()
            else : 
                self.skip_counter += 1
                if self.skip_counter >= 5 :
                    self.save_recording()

    def save_recording_new(self):
        if self.img_counter >= 1:   
            now = datetime.now()
            video_name = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}.avi'  
            out = cv2.VideoWriter(BUFFER_IMAGES_PATH+video_name, self.fourcc, self.fps, (640,480))

            # json file name
            json_fname = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}.json'
            
            print(video_name)            
            for image in self.temp_img_for_video : 
                out.write(image)
            log.info("Video crealog.info("")ted and saved -> "+video_name)
            
            # save json bbox
            json_file = open(json_fname, 'w')
            json.dump(self.temp_img_bbox_for_video, json_file)
            json_file.close()
            log.info("Video bbox JSON crealog.info("")ted and saved -> "+json_fname)

            # reset all
            self.temp_img_for_video.clear()
            self.temp_img_bbox_for_video.clear()
            self.img_counter = 0

    def save_recording(self):
        if self.img_counter >= 1:   
            now = datetime.now()
            video_name = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}.avi'  
            out = cv2.VideoWriter(BUFFER_IMAGES_PATH+video_name, self.fourcc, self.fps, (640,480))
            print(video_name)
            log.info("Video crealog.info("")ted and saved -> "+video_name)
            for image in self.temp_img_for_video : 
                out.write(image)

            self.temp_img_for_video.clear()
            self.img_counter = 0

    def start_new(self):
        log.info("Cam started functioning")
        while True :
            available, frame = self.cap.read()
            if available :
                self.start_storing_img_new(frame)
                #cv2.imshow("Motion Recorder",frame)
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    break

    def start(self):
        log.info("Cam started functioning")
        while True :
            available, frame = self.cap.read()
            if available :
                self.start_storing_img(frame)
                #cv2.imshow("Motion Recorder",frame)
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    break

    def end(self):
        #self.save_recording_new()
        #self.save_recording()
        self.cap.release()
        cv2.destroyAllWindows()

MR = MotionRecorder()
log.info("Object created")
#MR.start()
MR.start_new()
MR.end()
log.info("Script ended")
