

import time
import cv2
import argparse
import numpy as np
import os
import python_tele

# Short your code when run
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
# ap.add_argument('-c', '--config', required=True,
#                 help='/home/thanh/Documents/yolo/yolov.cfg')
# ap.add_argument('-w', '--weights', required=True,
#                 help='/home/thanh/Documents/yolo/yolov.weights')
# ap.add_argument('-cl', '--classes', required=True,
#                 help='/home/thanh/Documents/yolo/yolov.txt')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

#Save classes you want to detect
classes = None

with open('/home/thanh/Documents/yolo/yolov.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

'''
Blobs are multi-dimensional arrays that are used to represent images and other large binary 
objects in deep learning frameworks. They are used as the input to a deep learning model and they 
are typically passed through a series of layers in the model to make predictions
'''

#Use to load deeplearning model
net = cv2.dnn.readNet('/home/thanh/Documents/yolo/yolov.weights', '/home/thanh/Documents/yolo/yolov.cfg')

#Create a blob(standard of input deeplearning) with image
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

#Used to set the blob as the input of the network
net.setInput(blob)

#Make prediction on the input image
#get_output_layer(net): used to get the names of the output layers of the network
outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

# Thực hiện xác định bằng HOG và SVM
start = time.time()
# Save points,which is detected, result of above code
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

#Use to remove overlapping bounding boxes by applying a threshold to the confidence socres and non-maximun supperession algorithm
#This return the indices of the bounding boxes that were not suppressed
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
#Boxes a list of bounding box
#Confidence: a list of confidence scores for each bounding box
#conf_threshold: a threshold value for the confidence score. If bounding box with conf_score < conf_score than this thredshold will ve removed
#nms_threshold: value for NMS algorithm. This value controls how close together two bounding boxes can be before they are considerd to be overlapping

#Draw bounding boxes around object, which is detected
for i in indices:
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))



end = time.time()
print("YOLO Execution time: " + str(end-start))

cv2.imshow("object detection", image)
while (True):
    cv2.waitKey()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    break
cv2.imwrite("/home/thanh/Documents/Pandas/Detection_with_camera/object-detection.jpg", image)
python_tele.asyncio.run(python_tele.main('/home/thanh/Documents/Pandas/Detection_with_camera/object-detection.jpg'))

# print(img.path)
cv2.destroyAllWindows()