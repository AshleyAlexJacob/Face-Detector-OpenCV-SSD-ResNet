import numpy as np
import argparse
import cv2

def construct_argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required= True, type= str,
                    help= "Path to Input Image")
    ap.add_argument("-p", "--prototxt", required= True,
                    help= "Path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required= True,
                    help= "path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

    return vars(ap.parse_args())



# sourcery skip: simplify-numeric-comparison
if __name__ == "__main__":
    args =  construct_argument_parser()

    # load model
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # load image
    image = cv2.imread(args["image"])
    (h,w) = image.shape[:2]
    # resize image
    image_resized = cv2.resize(image, (300,300))
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, 
                                #  size of the blob
                                 (300,300), 
                                #  normalization factor (Mean Substraction from Each Channel)
                                 (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence> args["confidence"]:
            box = detections[0,0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box of the face along with the associated
		    # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY-10> 10 else startY+10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
    
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)


            




