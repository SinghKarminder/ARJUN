'''
@camera_type: Fixed Focus
@csv_type: minute mean data in csv per minute

@abstract: frames individually or .avi, mean count, csv .... 

@dscp:  this code creates csv every minute with mean_count of every seconds        
        also saves images in which movement is detected

@algo: basic bg subtractor, no GLCM, use RGB plane norm if required to remove shadow

@other: kills any other camera using programs to free busy cameras
        also, auto detect cam devices available to capture video

@packing: crop & pack bounding boxes, merge nearby

@author: vinay
'''

#use image crop
CROP_IMAGES = False

# merge nearby boxes
MERGE_NEARBY = True

# use this to display bounding boxes
FRAME_DEBUG = False
# allow 0 movement frames to be saved
ALLOW_NO_MOVEMENT_FRAME = False

# save CSV
SAVE_CSV = False

# use this to show on console when files are created
LOG_DEBUG = False

#us RGB plane normalization ?
USE_RGB_NORM = False

# max images per second (atmost 8-10 are possible per seconds)
# default 1 
MAX_IMAGES_PERSEC = 5

# save as .avi (video)
SAVE_AS_VIDEO = False
#---------------------

import cv2
import numpy as np
from datetime import datetime
import json
import logging as log
import csv
import subprocess
#pip install rectangle-packer
import rpack
from enum import Enum, auto

if LOG_DEBUG: print("Cam script started..")

with open(f"/etc/entomologist/ento.conf",'r') as file:
    data=json.load(file)

DEVICE_SERIAL_ID = data["device"]["SERIAL_ID"]
BUFFER_IMAGES_PATH = data["device"]["STORAGE_PATH"]
BUFFER_COUNT_PATH = data["device"]["COUNT_STORAGE_PATH"]

# general error codes
class CollatingErrorCodes(Enum):
    TOO_MANY_BOXES = auto(),
    MERGE_IMAGE_TOO_BIG = auto(),
    NO_MOTION = auto(),
    COLLATE_OK = auto()


class MotionRecorder(object):

    # for Fixedfocus
    VID_RESO, FPS = (1920,1080), 60
    #VID_RESO, FPS = (1920,1200), 55
    #VID_RESO, FPS = (1280,720), 120
    #VID_RESO, FPS = (1280,720), 60

    # for AutoFocus
    #VID_RESO, FPS = (640, 480), 60    
    #VID_RESO, FPS = (4208,3120), 9
    #VID_RESO, FPS = (4208,3120), 4.5
    #VID_RESO, FPS = (3840,2160), 15
    #VID_RESO, FPS = (3840,2160), 7.5
    #VID_RESO, FPS = (4096,2160), 7.5
    #VID_RESO, FPS = (1920,1080), 60
    #VID_RESO, FPS = (1920,1080), 30
    #VID_RESO, FPS = (1280,720), 60
    #VID_RESO, FPS = (1280,720), 30
    #VID_RESO, FPS = (640, 480), 120

    # the background Subractors
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    
    # don't exceed bounding boxes > 1/4 the area of actual resolution
    # even this big is unrealistic for a bee
    CONTOUR_AREA_LIMIT = VID_RESO[0]*VID_RESO[1]/4

    # skip a few frames for Background subtractor to get ready
    SKIP_FRAMES = 5

    # set offset (in pixels) for merging nearby boxes
    BOX_MERGE_MAX_DIST = 40

    # set the maximum number of boxes to allow collating
    MAX_BOX_COUNT_LIMIT = 50

    
    def _init_(self):
        # temp data structures
        self.img_mean_permin_list = []    
        self.img_count_sum = 0
        self.img_count = 0
        self.temp_img_for_video = []        
        self.last_minute = None
        self.last_second = None

        # video capture : from device
        self.cap = None

    def get_cam_deviceID(self, VID_RESO, FPS):
        '''checks for all connected camera devices and return the camera id
        vid0,vid1, etc. for the specified resolution
        
        -1 , if no deviceFound'''

        # fetch which ever camera is working
        for i in range(0,3):
            # first stop the device --if busy

            output = None
            pid = -1
            try:        
                output = subprocess.check_output(["fuser",f"/dev/video{i}"])                
            except:
                if LOG_DEBUG:
                    print(f"No working cam at: /dev/video{i}")
                pass
            
            if output:
                output = output.decode('utf-8')
                try:
                    pid = int(output)
                    # kill the busy process
                    subprocess.call(["kill","-9",f"{pid}"])
                except:
                    pass

            # start the device
            camera = cv2.VideoCapture(f"v4l2src device=/dev/video{i} ! video/x-raw, width={VID_RESO[0]}, \
                                      height={VID_RESO[1]}, framerate={FPS}/1, format=(string)UYVY ! decodebin \
                                      ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
            while True:
                success, frame = camera.read()  # read the camera frame
                camera.release()
                if not success:                
                    break
                else:
                    if LOG_DEBUG:
                        print(f"Found working cam at: /dev/video{i}")
                    return i

        return -1
        
    def process_img(self, frame):
        store = frame
        
        #--------------------------
        # RGB plane normalization
        if USE_RGB_NORM == True:
            rgb_planes = cv2.split(frame)

            result_norm_planes = []
            for plane in rgb_planes:
                dilated_img = cv2.dilate(plane, np.ones((11,11), np.uint8))
                bg_img = cv2.medianBlur(dilated_img, 5)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_norm_planes.append(norm_img)

            shadow_less_frame = cv2.merge(result_norm_planes) #result_norm

            # 120,255 works with some
            #  ---- use either - chnage THRES_MIN only
            THRES_MIN = 140
            #ret, thresh1 = cv2.threshold(shadow_less_frame,THRES_MIN,255,cv2.THRESH_BINARY)
            ret, thresh1 = cv2.threshold(shadow_less_frame,THRES_MIN, 255, cv2.THRESH_TOZERO_INV)

        
        #------------------         
        # bg sub & blurring - Noise removal       
        if USE_RGB_NORM == True:
            imgOut = MotionRecorder.subtractor.apply(thresh1)
        else:
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
        # need for closepacking of cropped images
        sizesRed = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)            
            x,y = x-5,y-5
            w,h = w+10,h+10

            aspectRatio = w/h if w>h else h/w

            if aspectRatio > 5: continue

            if MotionRecorder.CONTOUR_AREA_LIMIT >= area:                
                x = 0 if x < 0 else MotionRecorder.VID_RESO[0] if x > MotionRecorder.VID_RESO[0] else x
                y = 0 if y < 0 else MotionRecorder.VID_RESO[1] if y > MotionRecorder.VID_RESO[1] else y
                w = MotionRecorder.VID_RESO[0]-x if x+w > MotionRecorder.VID_RESO[0] else w
                h = MotionRecorder.VID_RESO[1]-y if y+h > MotionRecorder.VID_RESO[1] else h
                detectionsRed.append([x, y, w, h])
                sizesRed.append([w,h])
        
        contours, _ = cv2.findContours(imgOut3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        numberofObjects = 0
        
        detections = []
        # need for closepacking of cropped images
        sizes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)            
            x,y = x-5,y-5
            w,h = w+10,h+10

            aspectRatio = w/h if w>h else h/w

            if aspectRatio > 5: continue

            if MotionRecorder.CONTOUR_AREA_LIMIT >= area:
                # remove red countous
                enclosedRed = []
                enclosedSizesRed = []
                thisEnclosedCount = 0

                for cntR, szR in zip(detectionsRed, sizesRed):
                    _x, _y, = cntR[:2]

                    if (x < _x < x+w) and (y <_y <y+h):
                        thisEnclosedCount+=1
                        enclosedRed.append(cntR)
                        enclosedSizesRed.append(szR)

                if thisEnclosedCount < 2:
                    x = 0 if x < 0 else MotionRecorder.VID_RESO[0] if x > MotionRecorder.VID_RESO[0] else x
                    y = 0 if y < 0 else MotionRecorder.VID_RESO[1] if y > MotionRecorder.VID_RESO[1] else y
                    w = MotionRecorder.VID_RESO[0]-x if x+w > MotionRecorder.VID_RESO[0] else w
                    h = MotionRecorder.VID_RESO[1]-y if y+h > MotionRecorder.VID_RESO[1] else h
                    detections.append([x, y, w, h])
                    sizes.append([w,h])
                else:
                    detections.extend(enclosedRed)
                    sizes.extend(enclosedSizesRed)

        #------------------
        # display cntrs
        if FRAME_DEBUG:
            for cntr in detections:
                x,y,w,h = cntr
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 1)
                #cv2.drawContours(store, [cnt], -1, (255,0,0),2)
                numberofObjects = numberofObjects + 1
                cv2.putText(store,str(numberofObjects), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_8)

            cv2.putText(store,"Number of objects: " + str(len(detections)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_8)

        hasMovement = (len(detections) > 0)

        return hasMovement, frame, detections, sizes

    def merge_boxes(boxes, DIST):        
        merged_boxes = []
        sizes = []

        for box in boxes:
            # Create a temp copy of box
            new_box = box

            # check if this overlap with merged boxes
            overlaps = []
            for mb in merged_boxes:
                if (box[0] + box[2] + DIST >= mb[0] and mb[0] + mb[2] + DIST >= box[0] and
                    box[1] + box[3] + DIST >= mb[1] and mb[1] + mb[3] + DIST >= box[1]):
                    overlaps.append(mb)

            if len(overlaps) > 0:
                # merge all overlapping boxes into a single new merged box
                overlaps.append(box)
                new_box = (min([b[0] for b in overlaps]), min([b[1] for b in overlaps]),
                        max([b[0]+b[2] for b in overlaps])-min([b[0] for b in overlaps]),
                        max([b[1]+b[3] for b in overlaps])-min([b[1] for b in overlaps]))

                # Remove all overlapping boxes from the list of merged boxes
                merged_boxes = [mb for mb in merged_boxes if mb not in overlaps]

            # Add the new or merged box to the list of merged boxes
            merged_boxes.append(new_box)

            # get there sizes
            sizes = [[b[2],b[3]] for b in merged_boxes]

        return merged_boxes, sizes    

    def Collate(img, bboxes, sizes):
        '''Crop detected regions and merge as single image
        returns collatedImg, ErroCodes'''
        
        if len(bboxes) == 0:
            print("Got 0 boxes, won't save.")
            return img, CollatingErrorCodes.NO_MOTION
        
        if len(bboxes) > MotionRecorder.MAX_BOX_COUNT_LIMIT:
            print(f"Got a lot of boxes > {MotionRecorder.MAX_BOX_COUNT_LIMIT}, won't combine... won't save.")
            return img, CollatingErrorCodes.TOO_MANY_BOXES

        #get new positions        
        pos = rpack.pack(sizes)
        sizes = np.array(sizes)
        positions = np.array(pos)

        # get maxY and maxX ("req"uired for creating empty image)
        reqH = max(sizes+positions, key=lambda x: x[0])[0]
        reqW = max(sizes+positions, key=lambda x: x[1])[1]
        
        reqH = reqW = max(reqW, reqH)

        if reqH >= MotionRecorder.VID_RESO[0]:
            print("Exceeds original size after collating, saving original captured image.")
            return img, CollatingErrorCodes.MERGE_IMAGE_TOO_BIG

        newImg = np.zeros((reqH,reqW,3), np.uint8)
        
        for i in range(len(bboxes)):
            # copy pixels from img to newImg            
            y1, x1 = bboxes[i][0:2]
            h, l = sizes[i]
            y,x = positions[i]

            newImg[x:x+l,y:y+h] = img[x1:x1+l,y1:y1+h]
        
        return newImg, CollatingErrorCodes.COLLATE_OK

    def start_storing_img(self, img):
        if LOG_DEBUG: print('-----')
        
        hasMovement, img2, bbox, sizes = self.process_img(img.copy())
        
        errorCode = None
        debugImg = None
        countBoxesBeforeMerge = len(bbox)
        if FRAME_DEBUG:
            debugImg = img2
            if ALLOW_NO_MOVEMENT_FRAME : hasMovement = True
        
        if CROP_IMAGES:
            if MERGE_NEARBY:
                # merge nearby boxes
                if LOG_DEBUG: print("Merging boxcount:",len(bbox))
                merged_bboxes1 , _ = MotionRecorder.merge_boxes(bbox, MotionRecorder.BOX_MERGE_MAX_DIST)
                if LOG_DEBUG: print("After Merging...boxcount:",len(merged_bboxes1))

                # twice to merge new overlapping ones                
                merged_bboxes, sizes = MotionRecorder.merge_boxes( merged_bboxes1, 0 )
                if LOG_DEBUG: print("After Merging Overlaps...boxcount:",len(merged_bboxes1))

                bbox = merged_bboxes

            img, errorCode = MotionRecorder.Collate(img, bbox, sizes)

        # get current time
        now = datetime.now()

        # if change in minutes
        if self.last_minute == None:
            self.last_minute = now.minute
        elif self.last_minute != now.minute:
            # if sec changes, save last second count data
            mean_count_permin = 0
            if self.img_count != 0:
                mean_count_permin = self.img_count_sum // self.img_count
                
            min_str = str(self.last_minute).zfill(2)
            self.img_mean_permin_list.append((DEVICE_SERIAL_ID, now.strftime("%d-%m-%Y_%H-")+f'{self.last_minute}', mean_count_permin))

            if SAVE_CSV: self.save_csv(now)
            #reset all
            self.last_minute = now.minute
            self.img_mean_permin_list = []
            self.img_count_sum = 0
            self.img_count = 0

        
        if hasMovement and (self.img_count < MAX_IMAGES_PERSEC):

            if SAVE_AS_VIDEO:
                self.temp_img_for_video.append(img)
            else:
                # save image file
                self.temp_image_name = f'{now.strftime("%d-%m-%Y_%H-%M-%S-%f")}_{DEVICE_SERIAL_ID}.jpg'
                #self.save_recording(img)
                cv2.imwrite(BUFFER_IMAGES_PATH + self.temp_image_name, img)
                if LOG_DEBUG: print('Saved Image: ', self.temp_image_name, len(bbox))

            # assign count data                        
            self.img_count_sum += len(bbox)
            self.img_count += 1
        
        if self.last_second != now.second:
            self.last_second = now.second

            if SAVE_AS_VIDEO:
                self.save_recording(now)
        
    def save_recording(self, now):
        #now = datetime.now()
        video_name = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}.avi'
        out = cv2.VideoWriter(BUFFER_IMAGES_PATH + video_name, self.fourcc, self.FPS, self.VID_RESO)
                    
        for image in self.temp_img_for_video:
            out.write(image)

        # log data
        log.info("Video crealog.info("")ted and saved -> "+video_name)            
        if LOG_DEBUG: print('Saved Video: ', video_name)

        # reset all
        self.temp_img_for_video.clear()        
        self.img_counter = 0

    def save_csv(self, timeNow):
        # save as count_timeFrame_deviceID_countMeanInt.csv
        # save as count_DD-MM-YYYY_hh-mm_DOxxx_XXXX.csv

        csvName = f'count_{timeNow.strftime("%d-%m-%Y_%H")}-{self.last_minute}_{DEVICE_SERIAL_ID}.csv'
        with open(BUFFER_COUNT_PATH + csvName, 'w',  newline='') as csvFile:
            csvwriter = csv.writer(csvFile)
            # header
            csvwriter.writerow(["device_id","time_frame","insect_count"])        
            # data 
            csvwriter.writerows(self.img_mean_permin_list)
    
        log.info("Video bbox count CSV crealog.info("")ted and saved -> "+csvName)
        if LOG_DEBUG: print('CSV saved', csvName)
            
            
    def start(self):
        camID = -1
        while camID == -1:
            camID = self.get_cam_deviceID(MotionRecorder.VID_RESO, MotionRecorder.FPS)
        self.cap = cv2.VideoCapture(f"v4l2src device=/dev/video{camID} ! video/x-raw, width={MotionRecorder.VID_RESO[0]}, height={MotionRecorder.VID_RESO[1]}, framerate={MotionRecorder.FPS}/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
        if LOG_DEBUG: print("Cam started functioning")

        log.info("Cam started functioning")

        skipCount = 0
        
        while True :
            available, frame = self.cap.read()

            if available and skipCount > MotionRecorder.SKIP_FRAMES:
                self.start_storing_img(frame)
                # check exit
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    break
            else:
                skipCount+=1
                if FRAME_DEBUG:
                    print("...Device Unavailable")

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
# scp /home/tif-awadh/Desktop/gitCodes/ARJUN/Final_Codes/csvMeanOfMinutes/FixedFocusCamera/cam.py root@192.168.8.1:/usr/sbin/cam/cam.py
# scp /home/tif-awadh/Desktop/local_see3cam_test/device_code/microseconds/cam.py root@192.168.8.1:/usr/sbin/cam/cam.py
