import cv2 as cv
from datetime import datetime, timedelta
import json
import logging as log

log.basicConfig(filename='/var/tmp/cam.log', filemode='w', level=log.INFO, format='[%(asctime)s]- %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
log.info("Cam script started..")
with open(f"/etc/entomologist/ento.conf",'r') as file:
    data=json.load(file)

DEVICE_SERIAL_ID = data["device"]["SERIAL_ID"]
BUFFER_IMAGES_PATH = data["device"]["STORAGE_PATH"]

class ImageCapture:

    def ImageCompareFunction_1(newImg, lastImg):
        '''chooses the best image from the interval'''
        return newImg

    def __init__(self, camSource = "", timeIntevalMin = 1, ImageCompareFnc = None, capTimeValSec = 30):
        self._cap = camSource
        # minutes
        self._capTimeInt_min = timeIntevalMin

        # seconds in minutes
        self._capTimeVal_sec = capTimeValSec

        # function to compare images - Future 
        self._imgCompFnc = ImageCompareFnc

        # current Img in hand
        self._curImg = None

        self._onTime = datetime.now()
        self._lastMinDef = 0

    def start(self):

        self._onTime = datetime.now()

        log.info("Cam started functioning")

        while True :
            available, frame = self._cap.read()

            self._curImg = frame

            isTime = self.CheckTime()
            
            if isTime and available:
                self.SaveImage()

                # check exit
                if cv.waitKey(1) & 0xFF == ord('x'):
                    break

    def end(self):
        self.SaveImage()
        self._cap.release()
        cv.destroyAllWindows()


    def CheckTime(self):
        '''return whether to reset capture counter if timeInterval crossed
        isCaptureIntervalCrossed'''

        curTime=datetime.now()        
        
        _def = (curTime.hour*60 + curTime.minute) - (self._onTime.hour*60 + self._onTime.minute)

        if self._lastMinDef > _def:
            self._lastMinDef = _def
            print(_def)

        if (curTime.hour*60 + curTime.minute) - (self._onTime.hour*60 + self._onTime.minute) >= self._capTimeInt_min:
            #reset the time counter
            self._onTime = datetime.now()
            

            return True
        else:
            return False



    def SaveImage(self):
        '''save the caputred image at the given time'''
        try:
            if self._curImg == None:
                return
        except:
            pass

        now = datetime.now()
        img_name = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}.mjpg'
        cv.imwrite(BUFFER_IMAGES_PATH+img_name, self._curImg)

        log.info("Image Saved -> " + img_name)
        print(img_name)


#--------------

VID_RESO = (1280,720)
vid_cap = cv.VideoCapture(f"v4l2src device=/dev/video2 ! video/x-raw, width={VID_RESO[0]}, height={VID_RESO[1]}, framerate=60/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv.CAP_GSTREAMER)

imgCpt = ImageCapture(vid_cap, 1)

log.info("Object created")
imgCpt.start()
imgCpt.end()
log.info("Script ended")
