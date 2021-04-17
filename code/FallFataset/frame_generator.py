# this file transfer raw frames to continuous frame sequence
import os
import cv2

src_path = input("Please enter the src img set path: ")
target_path = input("Please enter the target path: ")
start_index = int(input("Start: "))
end_index = int(input("End: "))

ptr = len(os.listdir(target_path)) + 1
for i in range(end_index - start_index + 1 - 20):
    head_name = src_path + "\\rgb_"
    tail_name = src_path + "\\rgb_"
    head_index = start_index + i
    tail_index = start_index + i + 20
    for j in range(4-len(str(head_index))):
        head_name = head_name + "0"
    for j in range(4-len(str(tail_index))):
        tail_name = tail_name + "0"
    head_img = cv2.imread(head_name + str(head_index) + ".png")
    tail_img = cv2.imread(tail_name + str(tail_index) + ".png")

    head_target = target_path + "\\" + str(ptr) + ".png"
    cv2.imwrite(head_target, head_img)
    ptr = ptr + 1
    tail_target = target_path + "\\" + str(ptr) + ".png"
    cv2.imwrite(tail_target, tail_img)
    ptr = ptr + 1
