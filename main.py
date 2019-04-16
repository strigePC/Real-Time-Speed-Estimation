import cv2

import numpy as np
from skimage.measure import compare_ssim as ssim

from functions import transform_image, calculate_homography

# Global Variables
roi_h = 250
roi_w = 200
pts_src = np.array([[629, 361], [1066, 411], [396, 486], [946, 571]])
pts_dst = np.array([[0, 0], [roi_w - 1, 0], [0, roi_h - 1], [roi_w - 1, roi_h - 1]])
roi = np.zeros((roi_h, roi_w, 3), np.uint8)
cars = []
next_id = 0

# Reading a video
input_video = cv2.VideoCapture('input.mp4')

# Calculating homography
hom = calculate_homography(pts_dst, pts_src)

ret, road_empty = input_video.read()

roi_empty = np.zeros_like(roi)
transform_image(road_empty, roi_empty, hom)
roi_empty_gray = cv2.cvtColor(roi_empty, cv2.COLOR_BGR2GRAY)
roi_empty_gray = cv2.medianBlur(roi_empty_gray, 11)

# Working with each video frame
while input_video.isOpened():
    next_cars = []

    # Reading a frame from video
    ret, frame = input_video.read()
    if not ret:
        break

    transform_image(frame, roi, hom)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.medianBlur(roi_gray, 11)

    # Calculating the difference between images
    score, diff = ssim(roi_empty_gray, roi_gray, full=True)
    roi_diff = diff * 255
    np.clip(roi_diff, 0, 255, out=roi_diff)
    roi_diff = roi_diff.astype('uint8')

    # Threshold an image
    roi_thresh = cv2.threshold(roi_diff, 164, 255, cv2.THRESH_BINARY_INV)[1]
    roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8), iterations=2)

    # Getting the contours of an image
    cnts, hier = cv2.findContours(roi_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Drawing a bounding rectangle around cars
    for contour in cnts:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w * h > 1200:
            M = cv2.moments(contour)
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            print(cx, cy)
            new_car = {
                'id': next_id,
                'cx': cx,
                'cy': cy,
                'topLeft': (x, y),
                'bottomRight': (x + w, y + h),
                'speed': 0
            }

            for car in cars:
                if abs(car['cx'] - cx) <= 10 and abs(car['cy'] - cy) <= 30:
                    new_car['id'] = car['id']
                    speed = abs(car['cy'] - cy) * 9

                    if speed > 0:
                        new_car['speed'] = speed

            # Append a car for the next iteration
            next_cars.append(new_car)

            print(new_car['id'])

            # Increment Next Car ID if new car was added
            if new_car['id'] == next_id:
                next_id += 1

    cars = next_cars
    for car in cars:
        cv2.rectangle(roi, car['topLeft'], car['bottomRight'], (0, 255, 0), 1)

        if car['speed'] is not None:
            cv2.putText(roi, 'id: {}'.format(car['id']), (car['topLeft'][0], car['topLeft'][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.putText(roi, '{0:.0f} km/h'.format(car['speed']), (car['topLeft'][0], car['bottomRight'][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Displaying the playback
    # cv2.imshow('Input Video', frame)
    cv2.imshow('ROI', roi)
    cv2.imshow('ROI Empty Road', roi_empty_gray)
    cv2.imshow('ROI Gray', roi_gray)
    cv2.imshow('ROI Difference', roi_diff)
    cv2.imshow('ROI Threshold', roi_thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing and closing resources
input_video.release()
cv2.destroyAllWindows()
