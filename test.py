from ultralytics import YOLO
import cv2
import cvzone

#detects the head using the scalp smart model

def detect_head(person_image_path):

    model = YOLO('model/version4_nanoyolo_20epoch_best.pt')
    results = model.predict(person_image_path)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            return [x1,y1,x2,y2]
        
#pass the path of person image here
person_image_loc = 'test_images/b4.jpg'
person_image = cv2.imread(person_image_loc)
#pass the path of accessory image here
acc_image = cv2.imread('acc_images/hat1.png', cv2.IMREAD_UNCHANGED)


loc = detect_head(person_image_loc)
x1 = int(loc[0])
y1 = int(loc[1])
x2 = int(loc[2])
y2 = int(loc[3])

# x_resize = int(1.6*(x2-x1))
# y_resize = int(2.3*(y2-y1))

#THIS IS THE RESIZE FUNCTION WHICH CAN CHANGE AS PER THE TYPE OF HAT

x_resize = int(1.3*(x2-x1))
y_resize = int(1.3*(y2-y1))

resized_acc_image = cv2.resize(acc_image, (x_resize,y_resize))

# x_loc = int(0.72*x1)
# y_loc = int(-0.6*y1)

#THESE ARE THE STARTING X AND Y COORDINATES OF THE ACCESSORY

x_loc = int(x1/1.3 + (x2-x1)/11)
y_loc = int(y1/1.3 - (y2-y1)/3)

resultImg = cvzone.overlayPNG(person_image,resized_acc_image, [x_loc,y_loc])

cv2.imshow('result',resultImg)
cv2.waitKey(0)
cv2.destroyAllWindows()