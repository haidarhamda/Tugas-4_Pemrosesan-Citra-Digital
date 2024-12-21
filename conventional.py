import cv2
import numpy as np

cascade = cv2.CascadeClassifier('haar_cascade_car.xml')

# def get_edge(image: np.ndarray) -> ndarray:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     kernel = np.ones((3, 3), np.uint8)
#     open_segment = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#     sure_bg = cv2.dilate(open_segment, kernel, iterations=1)
#
#     dist_transform = cv2.distanceTransform(open_segment, cv2.DIST_L2, 5)
#
#     _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)
#
#     _, markers = cv2.connectedComponents(sure_fg)
#
#     markers = markers + 1
#     markers[unknown == 255] = 0
#     markers = cv2.watershed(image, markers)
#     image[markers == -1] = [0, 0, 255]
#
#     return image


def get_vehicle(image: np.ndarray) -> np.ndarray:
    global cascade

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vehicles = cascade.detectMultiScale(gray, 1.01, 1)

    for (x, y, w, h) in vehicles:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image


def get_vehicle_video(video: cv2.VideoCapture) -> None:
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow('Vehicle Detection', get_vehicle(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = cv2.VideoCapture('dataset_video1.avi')
    get_vehicle_video(video)
    # img = cv2.imread('car.jpg')
    # if img is None:
    #     exit(1)
    # # cv2.imshow('edge', get_edge(img))
    # cv2.imshow('Vehicle Detection', get_vehicle(img))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()