import cv2
from ultralytics import YOLO
import random

def yolo(image):
    yolo = YOLO('yolov8s.pt')
    r,g,b=random.randint(0,255),random.randint(0,255),random.randint(0,255)
    results=yolo.predict(image)
    for result in results:
        classes_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                scale = cls/2
                color = (min(max(int(r * scale), 0), 255), min(max(int(g * scale), 0), 255), min(max(int(b * scale), 0), 255))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img=image, text=f'{class_name} {box.conf[0]:.2f}', org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color, thickness=2)
    return image

if __name__ == "__main__":
    image_path = "vehicles.png"
    image = cv2.imread(image_path)
    if image is not None:
        processed_image = yolo(image)
        cv2.imshow("", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()