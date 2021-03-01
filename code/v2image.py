# This script transfers videos to image sequences
import cv2
import math
import os


def v2image(video_path, seq_path):
    """

    :param video_path:
    :param seq_path: the path of the folder of frame_seq of current video
    :return: none
    """
    folder = os.path.exists(seq_path)
    if not folder:
        os.makedirs(seq_path)

    cap = cv2.VideoCapture(video_path)
    frame_num = cap.get(7)

    flag, frame = cap.read()

    frame_cnt = 1
    while flag:
        path = seq_path + '\\' + str(frame_cnt) + '.png'
        cv2.imwrite(path, frame)
        frame_cnt = frame_cnt + 1
        flag, frame = cap.read()

    cap.release()


# v2image('video (1).avi', 'test_img_seq')

img_base_Path = 'img_seq'
# Coffee Room 01
videoPath = 'G:\毕业设计\FallDataset\Coffee_room_01\Coffee_room_01\Videos'
fileList = os.listdir(videoPath)
video_cnt = len(fileList)
for i in range(video_cnt):
    v_name = videoPath + '\\' + 'video (' + str(i+1) + ').avi'
    seq_name = img_base_Path + '\\' + 'CoffeeRoom01' + '\\' + 'video' + str(i+1)
    v2image(v_name, seq_name)
