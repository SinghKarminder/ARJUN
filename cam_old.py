# use this to show bounding box on image
DEBUG = True


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
    
    # video capture : from device
    cap = cv2.VideoCapture("v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480, framerate=60/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    #cap = cv2.VideoCapture("videotestsrc ! video/x-raw, format=I420, width=640, height=480 ! vpuenc_h264 ! appsink",cv2.CAP_GSTREAMER)

    # the background Subractors
    #subtractor = cv2.createBackgroundSubtractorMOG2()
    subtractor = cv2.createBackgroundSubtractorKNN()

    # FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')     # for windows
    fps = 60
    img_counter = 0    
    IMAGE_COUNTER_LIMIT = 100

    CONTOUR_AREA_LIMIT = 200
    
    temp_img_for_video = []
    temp_img_bbox_for_video = {}
    

    def _init_(self):
        pass


    def process_img(self, img):
        '''@vinay
        use Background Subtractor, Median Blur, Morph Close Filter
        to get bouding boxes
        
        return: 
        hasMovement : bool
        img : cvMat
        detections : list.list # bounding box x,y,w,h'''
        
        # are tested values - change if not relevant
        # out : processed image
        out = self.subtractor.apply(img)
        out = cv2.medianBlur(out, 5)
        kernel = np.ones((9, 9), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel,iterations = 4)

        # remove minute variations(gray 127) - consider only Major Movement (white patches 255)    
        out[out == 127] = 0 # 0-> black

        # get bounding boxes
        contours, _ = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > self.CONTOUR_AREA_LIMIT:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

                if DEBUG:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # if has boxes => valid frame else skip
        hasMovement = len(detections) > 0

        return hasMovement, img, detections


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
            out = cv2.VideoWriter(BUFFER_IMAGES_PATH+video_name, self.fourcc, self.fps, (640,480))
                        
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
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    break

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
