import numpy as np
import argparse
import cv2
from imutils.video import VideoStream
import imutils
import time

def construct_argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required= True,
                    help= "Path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required= True,
                    help= "path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

    return vars(ap.parse_args())

if __name__ == "__main__":
    args =  construct_argument_parser()

    # load model
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src= 0).start()
    time.sleep(2.0)

    # Loop Over Frames from video stream
    while True:
        # grab the frame from the threaded video stream and resize it
	    # to have a maximum width of 400 pixels
        
        frame  = vs.read()
        frame = imutils.resize(frame, width= 400)

        (h,w) = frame.shape[:2]

        target = (300,300)

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, target), 1.0, target, (104.0, 177.0, 123.0))

        # input the blob to network and run predictions
        net.setInput(blob)

        detections = net.forward()

        # draw boxes
        
        for i in range(detections.shape[2]):
            # confidence
            confidence = detections[0,0,i,2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence

            if confidence < args["confidence"]:
                continue
            
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0,0,i, 3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype('int')

            text = "{:2f}%".format(confidence * 100)
            y = startY-10 if startY-10 > 10 else startY+10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
            
            cv2.putText(frame, text , (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0, 255), 2)
        
        cv2.imshow("Realtime Face Detection", frame)
            
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()









