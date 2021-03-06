# This script transfers image sequence into clips, step = 20
import os
import cv2


seq_path = 'G:\Graduation-Design-Note\code\img_seq\CoffeeRoom01\\video3'
target_path = 'clips'
img_cnt = len(os.listdir(seq_path))
src_bias = 44
target_bias = 1

for i in range(img_cnt):
    if i + 21 + src_bias > img_cnt:
        break
    head_img = cv2.imread(seq_path + '\\' + str(i+1+src_bias) + '.png')
    tail_img = cv2.imread(seq_path + '\\' + str(i+21+src_bias) + '.png')
    cv2.imwrite(target_path + '\\' + str(target_bias) + '.png', head_img)
    target_bias = target_bias + 1
    cv2.imwrite(target_path + '\\' + str(target_bias) + '.png', tail_img)
    target_bias = target_bias + 1
