'''
@camera_type: Fixed Focus
@csv_type: mean of frame in 60 seconds in csv per minute


@abstract: frames individually, mean count, csv .... 

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

# use this to display bounding boxes
FRAME_DEBUG = False

# save CSV
SAVE_CSV = False

# use this to show on console when files are created
LOG_DEBUG = False

#us RGB plane normalization ?
USE_RGB_NORM = False

# max images per second (atmost 8-10 are possible per seconds)
# default 1 
MAX_IMAGES_PERSEC = 1
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

log.basicConfig(filename='/var/tmp/cam.log', filemode='w', level=log.INFO, format='[%(asctime)s]- %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
log.info("Cam script started..")
with open(f"/etc/entomologist/ento.conf",'r') as file:
    data=json.load(file)

DEVICE_SERIAL_ID = data["device"]["SERIAL_ID"]
BUFFER_IMAGES_PATH = data["device"]["STORAGE_PATH"]
BUFFER_COUNT_PATH = data["device"]["COUNT_STORAGE_PATH"]

class MotionRecorder(object):
    
    # for fixed focus
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

    # video capture : from device
    cap = None
    #cap = cv2.VideoCapture(f"v4l2src device=/dev/video2 ! video/x-raw, width={VID_RESO[0]}, height={VID_RESO[1]}, framerate={FPS}/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
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

    def get_cam_deviceID(self, VID_RESO, FPS):

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

            # get camera output
            camera = cv2.VideoCapture(f"v4l2src device=/dev/video{i} ! video/x-raw, width={VID_RESO[0]}, height={VID_RESO[1]}, framerate={FPS}/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
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

            if self.CONTOUR_AREA_LIMIT < area:
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
                    sizes.append([w,h])
                else:
                    detections.extend(enclosedRed)
                    sizes.extend(sizesRed)

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

        return hasMovement, frame, detections, sizes

    def merge_boxes(boxes, DIST):        
        merged_boxes = []

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
        '''Crop detected regions and merge as single image'''
        
        if len(bboxes) == 0 or len(bboxes) > 40:
            print("Got 0 or A lot of boxes, try combining, returning.... This takes a lot of time....")
            return img

        #get new positions
        #area = sum([i[0]*i[1] for i in sizes])
        #print(area)
        #dimArea = int(area**0.5) + 1
        pos = rpack.pack(sizes)# dimArea*2, dimArea*2)
        #print(type(pos))

        sizes = np.array(sizes)
        positions = np.array(pos)

        #print(positions)

        # get maxY and maxX 
        reqH = max(sizes+positions, key=lambda x: x[0])[0]
        reqW = max(sizes+positions, key=lambda x: x[1])[1]
        
        reqH = reqW = max(reqW, reqH)
        # no indent
        if reqH >= MotionRecorder.VID_RESO[0]:
            print("Exceeds original size, returning")
            return img

        #print('NewImageDims:',reqH, reqW)

        newImg = np.zeros((reqH,reqW,3), np.uint8)
        #newImg *= 255
        #print(newImg.shape)


        for i in range(len(bboxes)):
            # copy pixels from img to newImg
            #print('Cropping the bee...')
            #print(len(sizes), sizes)
            #print(len(positions) ,positions)
            #print(bboxes)
            #print('-'*50)
            y1, x1 = bboxes[i][0:2]
            h, l = sizes[i]

            y,x = positions[i]
            #print('PosOn_NewFrame:',x,x+l,y,y+h)
            #print('PosFrom_Frame :',x1,x1+l,y1,y1+h)
            #print(newImg.shape)
            #print(img.shape)
            newImg[x:x+l,y:y+h] = img[x1:x1+l,y1:y1+h]
        
        #cv2.imshow("ImgOut", newImg)
        
        return newImg

    def start_storing_img(self, img):
        
        hasMovement, img2, bbox, sizes = self.process_img(img)
        if FRAME_DEBUG:
            img = img2
            hasMovement = True
        
        elif CROP_IMAGES:
            # merge nearby boxes        
            merged_bboxes, _ = MotionRecorder.merge_boxes(bbox,MotionRecorder.BOX_MERGE_MAX_DIST)
            # twice to merge new overlapping ones
            merged_bboxes, sizes = MotionRecorder.merge_boxes( merged_bboxes, 0 )

            img = MotionRecorder.Collate(img, bbox, sizes)

        # get current time
        now = datetime.now()

        # if change in minutes
        if self.last_minute == None:
            self.last_minute = now.minute
        elif self.last_minute != now.minute:
            # if sec changes, save last second count data            
            if SAVE_CSV: self.save_csv(now)
            #reset all
            self.last_minute = now.minute
            self.img_mean_persec_list = []
            self.img_count_sum = 0
            self.img_count = 0

        
        if hasMovement and (self.img_count < MAX_IMAGES_PERSEC):

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
        camID = -1
        while camID == -1:
            camID = self.get_cam_deviceID(self.VID_RESO, self.FPS)

        self.cap = cv2.VideoCapture(f"v4l2src device=/dev/video{camID} ! video/x-raw, width={self.VID_RESO[0]}, height={self.VID_RESO[1]}, framerate={self.FPS}/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

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
