import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, copy_detected_face_only
from utils.torch_utils import select_device, load_classifier, time_synchronized

# ANU: add queue to route data from here to the FaceMatching process
from multiprocessing import Process, Queue
import threading
import numpy as np

# ANU: access SQL DB wrapper functions
import mysql.connector

# ANU: date time for time stamps
import datetime

# ANU: Bug fix: cannot use so much memory on single GPU with 2 CUDA modules.
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#ANU: Import DeepFace for FaceMatching
from deepface import DeepFace

#FaceMatchingModels = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
FaceMatchingModels = ["VGG-Face"]

# ANU: Color terminal prints
from termcolor import colored, cprint
import colorama
colorama.init()

# ANU: temp directory to store our stuff
# ANU: importing os module
import os
import sys
import shutil

TEMP_DIR_NAME = "temp"


# ANU: Interface with Twilio for SMS notifications
from twilio.rest import Client

###################################### SQL Interface functions ##########################################################
def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

def readBLOBFromEmployeePhotoTable():
    print("Reading BLOB data from EMPLOYEE photo table")

    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='maskitor',
                                             user='root',
                                             password='')

        cursor = connection.cursor()
        sql_fetch_blob_query = """SELECT * from employee_photo"""

        cursor.execute(sql_fetch_blob_query)
        record = cursor.fetchall()
            
        return record
            
    except mysql.connector.Error as error:
        print("Failed to read BLOB data from MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")



def insertDataInResultsTable(DetectedImagePATH, DetectedConfidenceLevel):
    print("Inserting CameraID, detected image and detection confidence into RESULTS table")
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='maskitor',
                                             user='root',
                                             password='')

        cursor = connection.cursor()
        sql_insert_blob_query = """ INSERT INTO results (CameraID, Capture_Date, Capture_Time, Confidence_level, Photo) VALUES (%s,%s,%s,%s,%s)"""

        ResultImage = convertToBinaryData(DetectedImagePATH)
        
        insert_blob_tuple = ("CAM001", datetime.date.today(), datetime.datetime.now().time(), float(DetectedConfidenceLevel), ResultImage)
        cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        connection.commit()
        
    except mysql.connector.Error as err:
        print(err)
        print("Error Code:", err.errno)
        print("SQLSTATE", err.sqlstate)
        print("Message", err.msg)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            
           
def insertIntoResultsEmployeeTable(EmpID):
    print("Inserting records to Results_Employee Table")
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='maskitor',
                                             user='root',
                                             password='')

        cursor = connection.cursor()
                   
        FetchResultIDQuery = """ SELECT ResultsID FROM results ORDER BY ResultsID DESC LIMIT 1"""
        cursor.execute(FetchResultIDQuery)
        ResultIDTuple = cursor.fetchall()
        ResultID = [x[0] for x in ResultIDTuple][0]
        
        sql_insert_resultID_query = """ INSERT INTO results_employee (ResultsID, EmployeeID) VALUES (%s,%s)"""
        insert_tuple = (ResultID, EmpID)

        result = cursor.execute(sql_insert_resultID_query, insert_tuple)
        connection.commit()
        

    except mysql.connector.Error as err:
        print(err)
        print("Error Code:", err.errno)
        print("SQLSTATE", err.sqlstate)
        print("Message", err.msg)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            
def fetchEmployeeDetails(EmpID):
    print("Fetching Employee Details from the Employee table")
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='maskitor',
                                             user='root',
                                             password='')

        cursor = connection.cursor()
                   
        FetchEmpDetails = """SELECT EmployeeID, First_Name, Last_Name, Phone FROM Employee where EmployeeID = %s"""
        cursor.execute(FetchEmpDetails,(EmpID,))
        EmpDetails = cursor.fetchall()
        
        return EmpDetails
       
      
    except mysql.connector.Error as err:
        print(err)
        print("Error Code:", err.errno)
        print("SQLSTATE", err.sqlstate)
        print("Message", err.msg)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            

def WhichCamerasAreONandWhere():
    print("Fetching Employee Details from the Employee table")
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='maskitor',
                                             user='root',
                                             password='')

        cursor = connection.cursor()
                   
        FetchCamStatus = """SELECT CameraID, Location FROM Camera where status = "open";"""
        cursor.execute(FetchCamStatus)
        CameraDetails = cursor.fetchall()
        
        return CameraDetails
       
      
    except mysql.connector.Error as err:
        print(err)
        print("Error Code:", err.errno)
        print("SQLSTATE", err.sqlstate)
        print("Message", err.msg)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
######################################### END SQL Interface functions ###############################################################            

def detect(FaceDetectorToFaceMatchingQ, save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
     
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = [[0,128,0],[0,0,255]] #facemask = "green", no-facemask = "red"
    
    # Let us create a TEMP directory here to store our temporary images and other MISC data
    # current_directory = os.getcwd()
    # temp_dir = os.path.join(current_directory, TEMP_DIR_NAME)
    
    # if os.path.exists(temp_dir):
        # shutil.rmtree(temp_dir)
        # print ("Previosuly existing temp work directory deleted")

    # os.mkdir(temp_dir)
    # print ("New temp work directory created")
   
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    CurrFrameCounter = 0
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                CurrFrameCounter = CurrFrameCounter + 1
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        dataN = det.cpu().numpy()[0]
                        #Write to the queue which feeds data to FaceMatching Process only when we have NO MASK situation
                        if ((dataN[5] == 1) and (CurrFrameCounter >= 30)):
                            
                            detected_face_only = copy_detected_face_only(xyxy, im0)
                            DataForQueue = [detected_face_only, dataN[4]]
                            # print ("Data for Q: ", DataForQueue)
                            # print ("CurrFrameCounter: ", CurrFrameCounter)
                            FaceDetectorToFaceMatchingQ.put(DataForQueue)
                            CurrFrameCounter = 0
                            
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        
            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    #print(f'Done. ({time.time() - t0:.3f}s)')

def FaceMatchingProcessFunc(FaceDetectorToFaceMatchingQ):

    # Let us create a TEMP directory here to store our temporary images and other MISC data
    current_directory = os.getcwd()
    temp_dir = os.path.join(current_directory, TEMP_DIR_NAME)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print ("Previosuly existing temp work directory deleted")

    os.mkdir(temp_dir)
    print ("New temp work directory created for FACEMatching")

    
    # Let us smartly read images from the employee table 
    SQLEmployeePhotoDBRecord = readBLOBFromEmployeePhotoTable()
    
    # Your Account Sid and Auth Token from twilio.com/console
    # and set the environment variables. See http://twil.io/secure
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    client = Client(account_sid, auth_token)

    
    # Let us build an array in memory from the SQL table of employee photos (much faster to convert SQL BLOBs into OpenCV arrays
    SQLDataInList = []
    for row in SQLEmployeePhotoDBRecord:
        print ("SQLTable Employee ID: ", row[0])
        SQLDataInList.append(row[0])
        
        img_np1 = np.asarray(bytearray(row[1]), dtype="uint8")
        img_np1 = cv2.imdecode(img_np1, cv2.IMREAD_COLOR)
        
        img_np2 = np.asarray(bytearray(row[2]), dtype="uint8")
        img_np2 = cv2.imdecode(img_np2, cv2.IMREAD_COLOR)
        
        img_np3 = np.asarray(bytearray(row[3]), dtype="uint8")
        img_np3 = cv2.imdecode(img_np3, cv2.IMREAD_COLOR)
            
        SQLDataInList.extend([img_np1, img_np2, img_np3])
        print ("SQLDataInList length is: ", str(len(SQLDataInList)))
        
    CameraOnLocationList = WhichCamerasAreONandWhere()
    CameraLocation = None
    
    if CameraOnLocationList:
        for row in CameraOnLocationList: 
            CameraLocation = row[1]   
           
        
    # Let us build VGG-Face model
    FaceMatchingModel = DeepFace.build_model('VGG-Face')
    
    DataFromQueue = None

    # let us parse the incoming FaceDetectorToFaceMatchingQ until program terminates
    while True:
        try:
            DataFromQueue = FaceDetectorToFaceMatchingQ.get(timeout=2.0)
            
            if (DataFromQueue):
                print("\r\n")
                cprint('Read new Image to match with employee DB from FaceDetectorToFaceMatchingQ', 'white', 'on_green')
                print("\r\n")
                
                #print("Current size of FaceDetectorToFaceMatchingQ: ", FaceDetectorToFaceMatchingQ.qsize())
                
                CameraImageToMatch = DataFromQueue[0].astype(np.uint8)
                cv2.imwrite(temp_dir + "\\CameraImageToMatch.png", CameraImageToMatch)
                
                # Let us insert detected face into SQL Results Table
                insertDataInResultsTable(temp_dir + "\\CameraImageToMatch.png", DataFromQueue[1])
                
                # Let us do some FACE matching of the unmasked person on the camera with the employee DB to see if we can send a notification
                CurrEmpIDImagesIterated = 0
                MatchFound = False
                    
                for index, element in enumerate(SQLDataInList):
                    if (((index % 4) == 0) or (index == 0)):
                        print ("Trying to see if current unmasked person is matching with Employee ID: ", element)
                        CurrEmpIDImagesIterated = element
                        continue

                    result = DeepFace.verify(CameraImageToMatch, element, model = FaceMatchingModel, enforce_detection=False, detector_backend="mtcnn")
                    print("\r\nIs EmployeeID: " + str(CurrEmpIDImagesIterated) + " found without a MASK ? ---> ", result["verified"])
                    print("\r\n")
                        
                    if (result["verified"] == True):
                        MatchFound = True
                        insertIntoResultsEmployeeTable(str(CurrEmpIDImagesIterated))
                        WithoutMaskEmployeeDetails = fetchEmployeeDetails(str(CurrEmpIDImagesIterated))
                            
                        for EmployeeDetailROW in WithoutMaskEmployeeDetails:
                            print("Employeeid = ", EmployeeDetailROW[0],) 
                            print("FirstName = ", EmployeeDetailROW[1])
                            print("LastName = ", EmployeeDetailROW[2])
                            print("Phone = ", EmployeeDetailROW[3])
                            
                        WithoutMaskMessageForAdmin = "[MASKITOR ALERT] Looks like our employee ID: " + str(EmployeeDetailROW[0]) + " NAME: " + str(EmployeeDetailROW[1]) + " " + str(EmployeeDetailROW[2]) + ", PHONE: +1" + str(EmployeeDetailROW[3]) + " is near CAM001, Location: " + str(CameraLocation) + ", without a MASK"
                        
                        WithoutMaskMessageForEmployee = "[MASKITOR ALERT] Sir/Madam, our CAM001, Location: " + str(CameraLocation) + " detected you without a MASK, your building access has been paused, contact SYS ADMIN @+1 (571)800-7940" 
                            
                        print("\r\n")
                        text = colored(WithoutMaskMessageForAdmin, 'red', 'on_grey')
                        print(text)
                        print("\r\n")
                        message = client.messages.create(body=WithoutMaskMessageForAdmin, from_='+17743077102', to='+15718007940')
                        print (message.sid)
                        message = client.messages.create(body=WithoutMaskMessageForEmployee, from_='+17743077102', to='+12026973972')
                        print (message.sid)
                        break
                            
                if (MatchFound == False):
                    WithoutMaskMessageForAdmin = "[MASKITOR ALERT] Looks like a NON employee (visitor) is near CAM001, Location: " + str(CameraLocation) + ", without a MASK....CATCH HIM"
                    print("\r\n")
                    cprint(WithoutMaskMessageForAdmin, 'white', 'on_red')
                    print("\r\n")
                    
                    message = client.messages.create(body=WithoutMaskMessageForAdmin, from_='+17743077102', to='+15718007940')
                    print (message.sid)
                 
                #let us suck the Q elements as fast as we can
                while True:
                    if FaceDetectorToFaceMatchingQ.qsize() == 0:
                        print ("FaceDetectorToFaceMatchingQ is vaccuumed")
                        break
                    DataFromQueueFLUSH = FaceDetectorToFaceMatchingQ.get()
          
            else:
                continue
        except:
            print("FaceDetectorToFaceMatchingQ is trying to fetch data for computation")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()
    
    CameraOnList = WhichCamerasAreONandWhere()
    if not CameraOnList:
        print("\r\n")
        cprint('No Camera is ON for MASKITOR to Run. HINT: Switch on some from the Camera Menu on the Admin Panel', 'blue', 'on_red')
        print("\r\n")
        for i in range(10,0,-1):
            print("Program exiting in " + str(i) + " seconds",end='\r', flush=True)
            time.sleep(1)
        sys.exit(0)
        
    
    #Face Matching process is launched here
    FaceDetectorToFaceMatchingQ = Queue(maxsize=50)
    
    FaceMatchingProcess = Process(target=FaceMatchingProcessFunc, args=(FaceDetectorToFaceMatchingQ,))
    FaceMatchingProcess.start()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(FaceDetectorToFaceMatchingQ)
                strip_optimizer(opt.weights)
        else:
            detect(FaceDetectorToFaceMatchingQ)
            
    FaceMatchingProcess.join()