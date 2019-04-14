import cv2

import numpy as np

# Global Variables
from functions import transform_image

roi_h = 250
roi_w = 200
pts_src = np.array([[629, 361], [1066, 411], [396, 486], [946, 571]])
pts_dst = np.array([[0, 0], [roi_w - 1, 0], [0, roi_h - 1], [roi_w - 1, roi_h - 1]])
roi = np.zeros((roi_h, roi_w, 3), np.uint8)

# Reading a video
input_video = cv2.VideoCapture('input.mp4')

# Calculating homography
hom, status = cv2.findHomography(pts_dst, pts_src)
# print(hom)

ret, road_empty = input_video.read()
roi_empty = transform_image(road_empty, roi, hom)
roi_empty_gray = cv2.cvtColor(roi_empty, cv2.COLOR_BGR2GRAY)

# Working with each video frame
while input_video.isOpened():
    # Reading a frame from video
    ret, frame = input_video.read()
    if not ret:
        break

    roi = transform_image(frame, roi, hom)
    cv2.imshow('Input Video', frame)
    cv2.imshow('ROI Empty Road', roi_empty)
    cv2.imshow('ROI', roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing and closing resources
input_video.release()
cv2.destroyAllWindows()
