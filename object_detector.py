import numpy as np
import argparse
import cv2

'''
script for running object detection in 20 classes:
python object_detector.py --prototxt deploy.prototxt --model MobileNetSSD_deploy.caffemodel --image john-arano-LzxsSWAVMYs-unsplash.jpg
python object_detector.py --prototxt SSD.prototxt --model MobileNetSSD_deploy.caffemodel --image car-g928bc75b5_1920.jpg
'''
# construct the argument parse and parse it
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="")
ap.add_argument("-p", "--prototxt", required=True, help="")
ap.add_argument("-m", "--model", required=True, help="")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="")
args = vars(ap.parse_args())

# 20 classes are classified for object detection in MobileNet SSD, and
# same number of colors of bounding box is automatically generated for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the pre-trained MobileNet SSD caffe model and prototxt locally
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# load the input image and construct an 4-dimention blob
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (h, w)), 0.007843, 
    (h, w), 120)
# pass the blob input through the network and obtain objects detection
print("[INFO] computing object detections...\n")
net.setInput(blob)
detections = net.forward()

# loop over the detected objects and print out on image with bounding boxes
for i in np.arange(0, detections.shape[2]):
    # extract the confidence associated with each detected object
    # filter out the weak predictions by confidence threshold defined ahead
    confidence = detections[0, 0, i, 2]
    if confidence > args["confidence"]:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # print out prediction result on the console
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
        print("[INFO]", format(label))
        # draw bounding box for a detected object,
        # and put a label on the left top side of box
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the processed image
cv2.imshow("output", image)
cv2.waitKey(0)

'''
cv2.dnn.blobFromImage()
https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
Mat cv::dnn::blobFromImage	(	
    InputArray 	image,
    double 	scalefactor = 1.0,
    const Size & 	size = Size(),
    const Scalar & 	mean = Scalar(),
    bool 	swapRB = false,
    bool 	crop = false,
    int 	ddepth = CV_32F 
)	

cv2.dnn.Net.forward()
https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#a98ed94cb6ef7063d3697259566da310b

'''