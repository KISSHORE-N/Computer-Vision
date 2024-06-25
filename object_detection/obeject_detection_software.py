import cv2

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

classes = []

with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("object list")
print(classes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:
    ret, frame = cap.read()

    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
        cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (0, 0, 255), 3)
        
    print("class ids", class_ids)
    print("scores", scores)
    print("bbboxes", bboxes)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

