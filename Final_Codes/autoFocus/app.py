from itertools import count
import json
from multiprocessing import dummy
from urllib import response
from flask import Flask, render_template, Response, redirect, request, session, url_for, abort, send_file, jsonify
import cv2
import time
import os
import subprocess

app = Flask(__name__)
app.config['SECRET_KEY']="asdadvadfsdfs"      #random secret key
app.config['ENV']='development'
app.config['UPLOAD_FOLDER']='/media/mmcblk1p1'
app.config['RANA_FOLDER']='/usr/sbin/rana'
app.config['RANA_CONFIG_PATH'] = '/usr/sbin/rana/ranacore.conf'
app.config['credentials'] = '/usr/sbin/device-manager/DeviceManager/credentials.json'

def readFile(fileName):
    path="/tmp/"+fileName
    data={}
    try:
        with open(path ,'r') as file:
            data=json.load(file)
    except FileNotFoundError:
        data={"error":"File not found"}
    except json.decoder.JSONDecodeError:
        data={"error":"File is passed instead of json file"}
    except Exception as e:
        data={"error":str(e)}
    return data

def readRanaConfigData():
    temp=[]
    with open(app.config['RANA_CONFIG_PATH'],'r') as file:
        data=file.readlines()
        for line in data:
            if line[0]!='#' and line[0]!='\n':
                ind=line.index(" ")
                key=line[:ind]
                value=line[ind+1:]
                temp.append([key,value])
    return temp
        

def readData():
    subprocess.call("/usr/sbin/device-manager/DeviceManager/job_data.sh")
    time.sleep(0.05)

    subprocess.call("/usr/sbin/device-manager/DeviceManager/cellular.sh")
    time.sleep(0.05)

    subprocess.call("/usr/sbin/device-manager/DeviceManager/storage_state.sh")
    time.sleep(0.05)


    data={}
    tmp=readFile("devicestats")
    if "error" in tmp:
        data={
            "cpuInfo":{"usage":tmp["error"]},
            "gpuInfo":{"memoryUsage":404},
            "internet":{"connectivity":tmp["error"],"signal":tmp["error"]},
            "ramInfo":{"total":tmp["error"],"usage":tmp["error"],"free":tmp["error"]},
            "generalInfo":{"board_serial":tmp["error"],"board_type":"NRF","board_revision":tmp["error"]}
        }
    else:
        data=tmp

    tmp=readFile("met")
    if "error" in tmp:
        data['temperature']={"Relative_humidity":tmp["error"],"Temperature_c":tmp["error"],"Temperature_f":tmp["error"]}
    else:
        data['temperature']=tmp

    tmp=readFile("battery_parameters")
    if "error" in tmp:
        data['battery_parameters']={"Voltage":tmp["error"],"Internal_temperature":tmp["error"],"Average_current":tmp["error"]}
    else:
        data['battery_parameters']=tmp

    tmp=readFile("light_intensity")
    if "error" in tmp:
        data['light_intensity']={"Light_Intensity":tmp["error"]}
    else:
        data['light_intensity']=tmp

    tmp=readFile("gps")
    if "error" in tmp:
        data['gps']={"location":{"longitude":tmp["error"],"latitude":tmp["error"],"altitude":tmp["error"]}}
    else:
        data['gps']=tmp

    tmp=readFile("job")
    #print(tmp)
    if "error" in tmp:
        data['job']={"status":tmp["error"],"id":tmp["error"],"start_time":tmp["error"],"end_time":tmp["error"]}
    else:
        data['job']=tmp
        #print(data['job'])
                
    tmp=readFile("storage")
    #print(tmp)
    if "error" in tmp:
        data['storage']={"total":tmp["error"],"used":tmp["error"],"free":tmp["error"],"file":tmp["error"]}
    else:
        data['storage']=tmp


    tmp=readFile("cellular")
    #print(tmp)
    if "error" in tmp:
        data['cellular']={"operator":tmp["error"],"strength":tmp["error"],"state":tmp["error"],"pow":tmp["error"],"reg":tmp["error"],"tech":tmp["error"],"op_id":tmp["error"],"imei":tmp["error"],"apn":tmp["error"]}
    else:
        data['cellular']=tmp

    #dbg=subprocess.call(["cat","/tmp/debug_modem"])
    dbg=subprocess.Popen(["cat","/tmp/debug_modem"],stdout=subprocess.PIPE)    
    dbg=dbg.stdout.readlines()
                
    print(dbg)      
    return data,dbg

# update camera controls
def updateData(keyValue):
        data={}
        path="/etc/entomologist/"
        with open(path + "camera_control.conf",'r') as file:
            data=json.load(file)
        with open(path + "camera_control.conf",'w') as file:
            data.update(keyValue)
            #data.update({name:dataa})
            json.dump(data,file,indent=4,separators=(',', ': '))



@app.route('/upd')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        #f.save(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        return render_template("success.html", name = f.filename)  

@app.route('/',methods=["GET","POST"])
def login():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    if request.method=="POST":
        email=request.form.get('email')
        password=request.form.get('pass')
        credentials=None
        with open(app.config['credentials']) as file:
            credentials=json.load(file)
        if credentials['email']==email and credentials['password']==password:
            session['username']=credentials['username']
            return redirect(url_for('dashboard'))
    return render_template('login.html')

def get_cam_deviceID(VID_RESO, fps):

    # fetch which ever camera is working
    for i in range(0,3):
        # first stop the device --if busy

        output = None
        pid = -1
        try:        
            output = subprocess.check_output(["fuser",f"/dev/video{i}"])                
        except:
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
        camera = cv2.VideoCapture(f"v4l2src device=/dev/video{i} ! video/x-raw, width={VID_RESO[0]}, height={VID_RESO[1]}, framerate={fps}/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
        while True:
            success, frame = camera.read()  # read the camera frame
            camera.release()
            if not success:                
                break
            else:
                return i

    return 2

def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture("/usr/sbin/device-manager/DeviceManager/render.mp4")  # use 0 for web camera
    #camera.set(cv2.CAP_PROP_FPS,120)
    while os.path.exists('/tmp/rana_active'):
        time.sleep(0.05)
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    camera.release()
    subprocess.call(["systemctl","stop","rana"])
    subprocess.call(["systemctl","stop","cam"])

    # for fixed focus

    #VID_RESO, fps = (1920,1080), 60
    #VID_RESO, fps = (1920,1200), 55
    #VID_RESO, fps = (1280,720), 120
    #VID_RESO, fps = (1280,720), 60

    #for autofocus
    VID_RESO, fps = (640, 480), 60     
    # video capture : from device

    camID = get_cam_deviceID(VID_RESO, fps)
    camera = cv2.VideoCapture(f"v4l2src device=/dev/video{camID} ! video/x-raw, width={VID_RESO[0]}, height={VID_RESO[1]}, framerate={fps}/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    #camera = cv2.VideoCapture(2)  # use 0 for web camera
    #camera.set(cv2.CAP_PROP_FPS,60)
    #camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    #camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    #  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
    # for local webcam use cv2.VideoCapture(0)
    while True:
        #print("kuch toh")
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            #cv2.rectangle(frame,(100,100),(400,400),(255,0,0),2)
            #frame = cv2.flip(frame, 0)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        


@app.route('/video_feed')
def videoFeed():
    if 'username' in  session:
    #Video streaming route. Put this in the src attribute of an img tag
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return redirect(url_for('login'))

@app.route('/video')
def video(): 
    if 'username' in session:
        data={}
        try:
            var = subprocess.check_output("v4l2-ctl --device /dev/video2 --list-ctrls".split())
            output = var.decode('utf-8')
            output = output.split('\n')
            for index in range(len(output)-1):
                output[index] = output[index].strip()
                temp = output[index].split()
                if '(int)' in temp:
                    data[temp[0]]=[]
                    data[temp[0]].append(temp[4].split('=')[-1])
                    data[temp[0]].append(temp[5].split('=')[-1])
                    data[temp[0]].append(temp[6].split('=')[-1])
                    data[temp[0]].append(temp[7].split('=')[-1])
                    data[temp[0]].append(temp[8].split('=')[-1])
        except:
            data["error"]="Something is wrong on v4l2"
        return render_template('videoFeed.html',data=data)
    return redirect(url_for('login'))

@app.route('/setCamControls')
def setCamControls():
    if 'username' in session:
        args = request.args
        key = args.get('key')
        value = args.get('value')
        try:
            subprocess.call(f"v4l2-ctl --device /dev/video2 --set-ctrl={key}={value}".split())
            updateData({key:value})
            resp = {'msg':'success'}
            return  jsonify(resp)
        except:
            resp = {'msg':'error'}
            return  jsonify(resp)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        dummyData={
            "cpuInfo":{"usage":5.6},
            "gpuInfo":{"memoryUsage":2.3},
            "light_intensity":{"Light_Intensity":6.6},
            "internet":{"connectivity":True,"signal":2.3},
            "ramInfo":{"total":4,"usage":3,"free":1},
            "gps":{"location":{"longitude":1,"latitude":2,"altitude":3}},
            "temperature":{"Relative_humidity":32,"Temperature_c":21,"Temperature_f":37},
            "battery_parameters":{"Voltage":2.5,"Internal_temperature":38,"Average_current":2.7},
            "generalInfo":{"board_serial":34534,"board_type":"NRF","board_revision":2.3}
        }
        dat,db=readData()
        print(db)
        return render_template('Dashboard.html',data=dat,dbg=db)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    if 'username' in session:
        session.pop('username')
    return redirect(url_for('login'))

@app.route('/files')
def files():
    path="/media/mmcblk1p1/upload/"  #path for the directory's of file
    if not os.path.exists(path):
        return abort(404)

    files = os.listdir(path)
    return render_template('files.html',files=files)

@app.route('/files/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    dir="/media/mmcblk1p1/upload/"+filename  #path for the directory's of file
    # Returning file from appended path
    return send_file(dir)

@app.route('/configurations')
def configurations():
    data=readRanaConfigData()
    return render_template('configurations.html',data=data)

@app.route('/saveRanaConfig',methods=['POST'])
def saveRanaConfig():
    if 'username' in session and request.method == 'POST':
        formData=request.form
        contentDict={}
        content=None
        with open(app.config['RANA_CONFIG_PATH'],'r') as file:
            content=file.readlines()
        count=0
        for line in content:
            if line[0]!='#' and line[0]!='\n':
                ind=line.index(" ")
                key=line[:ind]
                value=formData[key]
                contentDict[key]=(value,count)
            count+=1
        for key in contentDict:
            line=key+" "+contentDict[key][0]+'\n'
            content[contentDict[key][1]]=line
        with open(app.config['RANA_CONFIG_PATH'],'w') as file:
            file.writelines(content)
    return redirect(url_for('configurations'))

@app.route('/configurations/file', methods=['GET', 'POST'])
def downloadConfFile():
    dir="/usr/sbin/rana/ranacore.conf"  #defing the path for conf file
    # Returning file from appended path
    return send_file(dir)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        #f.save(f.filename)  #path where the file has to be saved
        f.save(os.path.join(app.config['RANA_FOLDER'], f.filename))
        return 'file uploaded successfully'
    return 'Something went wrong'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)


'''
scp /home/tif-awadh/Desktop/local_see3cam_test/ResoTest_Fixed_Focus_nightlybuilds/app.py root@192.168.8.1:/usr/sbin/device-manager/DeviceManager/app.py
'''
