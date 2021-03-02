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


def v_processor(videoPath, framePath, bias):
    """

    :param bias:
    :param videoPath:
    :param framePath:
    :return: none
    """
    img_base_Path = 'img_seq'
    fileList = os.listdir(videoPath)
    video_cnt = len(fileList)
    for i in range(video_cnt):
        v_name = videoPath + '\\' + 'video (' + str(i+bias) + ').avi'
        seq_name = img_base_Path + '\\' + framePath + '\\' + 'video' + str(i+1)
        v2image(v_name, seq_name)


v_path = input("Please enter the path of video sets:")
frame_path = input("Please enter the target path of frame sets:")
cnt_bias = int(input("Please enter the bias:"))
v_processor(v_path, frame_path, cnt_bias)
