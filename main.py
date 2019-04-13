import cv2

# Reading a video
input_video = cv2.VideoCapture('input.mp4')

# Working with each video frame
while input_video.isOpened():
    # Reading a frame from video
    ret, frame = input_video.read()
    if not ret:
        break

    cv2.imshow('Input Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing and closing resources
input_video.release()
cv2.destroyAllWindows()
