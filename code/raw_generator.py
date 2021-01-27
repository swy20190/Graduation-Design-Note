# This file generates raw joint coordinate data
from img_parser import parse
import cv2
import numpy as np
import pandas as pd
import os

protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"
# Read the network into memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
nPoints = 18

ADLImgNum = len(os.listdir("images/ADL"))
# print(ADLImgNum)
ADLRawList = []
for i in range(ADLImgNum):
    print("processing " + str(i) + "\n")
    image_path = "images/ADL/ADL_" + str(i+1) + ".png"
    tmp_line = parse(net, nPoints, image_path)
    flat_line = []
    for coord in tmp_line:
        if coord:
            flat_line.append(coord[0])
            flat_line.append(coord[1])
        else:
            flat_line.append(None)
            flat_line.append(None)
    ADLRawList.append(flat_line)

ADLRawDf = pd.DataFrame(ADLRawList, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
                                             '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x', '11y',
                                             '12x', '12y', '13x', '13y', '14x', '14y', '15x', '15y', '16x', '16y',
                                             '17x', '17y'])
# write the data into csv file
ADLRawDf.to_csv("data/ADLRaw.csv", encoding="utf-8")

FallImgNum = len(os.listdir("images/fall"))
FallRawList = []

for i in range(FallImgNum):
    print("processing " + str(i) + "\n")
    image_path = "images/fall/fall_" + str(i+1) + ".png"
    tmp_line = parse(net, nPoints, image_path)
    flat_line = []
    for coord in tmp_line:
        if coord:
            flat_line.append(coord[0])
            flat_line.append(coord[1])
        else:
            flat_line.append(None)
            flat_line.append(None)
    FallRawList.append(flat_line)

FallRawDf = pd.DataFrame(FallRawList, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
                                               '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x',
                                               '11y', '12x', '12y', '13x', '13y', '14x', '14y', '15x', '15y', '16x',
                                               '16y', '17x', '17y'])

FallRawDf.to_csv("data/FallRaw.csv", encoding="utf-8")
