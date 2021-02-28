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


v2image('video (1).avi', 'test_img_seq')
