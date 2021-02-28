# This script transfers videos to image sequences
import cv2
import math

cap = cv2.VideoCapture('video (1).avi')
frame_num = cap.get(7)

frame_width = math.ceil(cap.get(3))
frame_height = math.ceil(cap.get(4))
frame_fps = math.ceil(cap.get(5))

flag, frame = cap.read()


frame_cnt = 1
while flag:
    path = 'img_seq_test\img_' + str(frame_cnt) + '.png'
    cv2.imwrite(path, frame)
    frame_cnt = frame_cnt+1
    flag, frame = cap.read()

cap.release()
