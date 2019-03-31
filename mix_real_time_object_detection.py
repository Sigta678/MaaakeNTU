# USAGE
# python mix_real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

import paho.mqtt.publish as publish
import boto3

def index_faces(bucket, key, collection_id, attributes=(), region="us-east-1"):
    rekognition = boto3.client("rekognition", region)
    response = rekognition.index_faces(Image={"S3Object": {"Bucket": bucket,"Name": key,}},CollectionId=collection_id,DetectionAttributes=attributes,)
    return response['FaceRecords']
def init_track(bucket, photo, collection_id):
    x=[]
    y=[]
    track=[]
    for record in index_faces(bucket, photo, collection_id):
        face = record['Face']
        details = record['FaceDetail']
        #print(details)
        #print("  FaceId: {}".format(face['FaceId']))
        x.append(details['BoundingBox']['Left'] + details['BoundingBox']['Width']/2)
        y.append(details['BoundingBox']['Top'] - details['BoundingBox']['Height']/2)
        track.append(face['FaceId'])
    return x,y,track
def update_track(bucket, photo, collection_id, track_name):
    indx=0
    count=0
    all_names=[]
    nx=[]
    ny=[]
    track_x, track_y = 0 , 0
    for record in index_faces(bucket, photo, collection_id):
        face = record['Face']
        details = record['FaceDetail']
        #print(details)
        #print("  FaceId: {}".format(face['FaceId']))
        all_names.append(face['FaceId'])
        nx.append(details['BoundingBox']['Left'] + details['BoundingBox']['Width']/2)
        ny.append(details['BoundingBox']['Top'] - details['BoundingBox']['Height']/2)
        if(face['FaceId'] == track_name):
            track_x = details['BoundingBox']['Left'] + details['BoundingBox']['Width']/2
            ##### Might be minus??!! #####
            track_y = details['BoundingBox']['Top'] - details['BoundingBox']['Height']/2
            indx = count
        count = count+1
    if(len(all_names)!=0):
        return track_x, track_y, all_names[indx], nx, ny, all_names
    else:
        return track_x, track_y, [], nx, ny, all_names
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# previous point
prev_X=[]
prev_Y=[]
now_X=[]
now_Y=[]
tmp_dist=[]
dist=0
ind=0
dist_X=0
dist_Y=0
track_X=0
track_Y=0
init_name=[]
track_name=''
new_track_names=[]
# ind=0
# S3
bucket='track-bucket'
#photo='person3.png'
COLLECTION = "my-collection-id"

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

# Init aws
frame = vs.read()
cv2.imwrite('init3.png',frame)
photo = 'init3.png'
#Upload files
s3 = boto3.client('s3')
s3.upload_file(photo, bucket, photo)
prev_X, prev_Y, init_name = init_track(bucket, photo, COLLECTION)
num = len(init_name)
publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78b/bar", num, hostname="broker.hivemq.com")
for i in range(len(init_name)):
    info = str(prev_X[i]) + ',' + str(prev_Y[i]) + ',' + str(init_name[i])
    # Publish for init names
    publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78b/bar", info, hostname="broker.hivemq.com")

# loop over the frames from the video stream
timeFrame = 50
timeFrame2=20
c = 1
while True:

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            midX = (startX + endX)/2
            midY = (startY + endY)/2
            
            now_X.append(midX)
            now_Y.append(midY)            
            
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            new_track_names.clear()
            label = label.split(':')
            new_track_names.append(label[0])
            print(new_track_names)
            print("\n")
                
    for i in range(len(now_X)):
        tmp_dist.append((track_X - now_X[i])**2 + (track_Y - now_Y[i])**2)

    if len(tmp_dist) != 0:
        dist=tmp_dist[0]
    
    ind = 0
    if(len(now_X)!=0):
        for j in range(len(now_X)-1):
            if(dist > tmp_dist[j+1]):
                dist = tmp_dist[j+1]
                ind = j+1
    # print(len(now_X),ind)
    if len(now_X) != 0 :
        # print(now_X)
        # Move the arm
        dist_X = now_X[ind] - track_X
        dist_Y = now_Y[ind] - track_Y
        # The point for next loop to track and calc distance
        track_X = now_X[ind]
        track_Y = now_Y[ind]
        
        # Update values
        prev_X = now_X
        prev_Y = now_Y
        # Clear the values that will be append next loop
        now_X.clear()
        now_Y.clear()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Update using awsAPI
    if(c%timeFrame == 0):
        # Track name在這裡更新之前的需求 (Arm part?)
        
        # hello = []
        # Publish for update track_names
        #publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78b/bar", len(new_track_names), hostname="broker.hivemq.com")
        for k in range(len(new_track_names)):
            # 這裡的info已經亂掉了, 必須全部重新拿
            try:
                info = str(now_X[k])+','+str(now_Y[k])+','+str(new_track_names[k])
                publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78b/bar", info, hostname="broker.hivemq.com")
            except:
                print("index out of range in line 223")
        track_X, track_Y, track_name, now_X, now_Y, new_track_names = update_track(bucket, photo, COLLECTION, track_name)
    if(c%timeFrame2 == 0):
        tname = str(track_X)+','+str(track_Y)+','+ str(track_name)
        publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78b/bar", 1, hostname="broker.hivemq.com")
        publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78b/bar", tname, hostname="broker.hivemq.com")
        #publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78b/bar", track_X, hostname="broker.hivemq.com")
        #publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78b/bar", track_Y, hostname="broker.hivemq.com")
        #publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78c/bar", "target_position", hostname="broker.hivemq.com")
        ttrack = str(track_X)+','+str(track_Y)
        publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78c/bar", ttrack, hostname="broker.hivemq.com")
        #publish.single("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78c/bar", track_Y, hostname="broker.hivemq.com")

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    c = c+1
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()