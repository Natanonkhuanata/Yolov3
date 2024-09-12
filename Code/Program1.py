import numpy as np
import cv2


#print(cv2.__version__)


scale = 0.75
img_size = (320,320)

cap = cv2.VideoCapture(0)


net = cv2.dnn.readNet("Code/yolov3.weights","Code/yolov3.cfg")
    
classes = []
class_ids = []
confidences = []
boxes = []
def locateObject(Output,img): 
    with open("Code/coco.names", "r") as f:
        classes = f.read().splitlines()



    for out in Output:
       for detection in out:
           score = detection[5:]
           class_id = np.argmax(score)
           confidence = score[class_id]

           if confidence > 0.5:
               center_x = int(detection[0] * width)
               center_y = int(detection[1] * height)
               w = int(detection[2] * width)
               h = int(detection[3] * height)

               x = int(center_x - w/2)
               y = int(center_y - h/2)

               boxes.append([x,y,w,h])
               confidences.append(float(confidence))
               class_ids.append(class_id)

    indexs = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4) #วาดกรอบสี่เหลี่ยน x,y,w,h
    colors = np.random.uniform(0,255,size=(len(boxes),3))
    font = cv2.FONT_HERSHEY_SIMPLEX


    for i in range(len(boxes)):
        if i in indexs:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+w),color,2)
            cv2.putText(img,label+" "+confidence,(x,y+20),font,2,(255,255,255),2)


while (cap.isOpened()):
    success, img = cap.read()
    img = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    
    height , width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img , 0.00392 , img_size, (0,0,0) , crop=False)
    Output_layers = net.getUnconnectedOutLayersNames()
    net.setInput(blob)
    Output = net.forward(Output_layers)
    locateObject(Output,img)

    cv2.imshow("Images",img)
    if cv2.waitKey(1) == 27:
        break
   


cap.release()
cv2.destroyAllWindows()