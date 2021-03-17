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

TrainImgNum = len(os.listdir("clips"))
# print(ADLImgNum)
TrainRawList = []
for i in range(TrainImgNum):
    print("processing " + str(i) + "\n")
    image_path = "clips/" + str(i+1) + ".png"
    tmp_line = parse(net, nPoints, image_path)
    flat_line = []
    for coord in tmp_line:
        if coord:
            flat_line.append(coord[0])
            flat_line.append(coord[1])
        else:
            flat_line.append(None)
            flat_line.append(None)
    TrainRawList.append(flat_line)

TrainRawDf = pd.DataFrame(TrainRawList, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
                                             '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x', '11y',
                                             '12x', '12y', '13x', '13y', '14x', '14y', '15x', '15y', '16x', '16y',
                                             '17x', '17y'])
# write the data into csv file
TrainRawDf.to_csv("data/clipTrainRaw.csv", encoding="utf-8")

TestImgNum = len(os.listdir("clips_test"))
TestRawList = []

for i in range(TestImgNum):
    print("processing " + str(i) + "\n")
    image_path = "clips_test" + str(i+1) + ".png"
    tmp_line = parse(net, nPoints, image_path)
    flat_line = []
    for coord in tmp_line:
        if coord:
            flat_line.append(coord[0])
            flat_line.append(coord[1])
        else:
            flat_line.append(None)
            flat_line.append(None)
    TestRawList.append(flat_line)

FallRawDf = pd.DataFrame(TestRawList, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
                                               '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x',
                                               '11y', '12x', '12y', '13x', '13y', '14x', '14y', '15x', '15y', '16x',
                                               '16y', '17x', '17y'])

FallRawDf.to_csv("data/clipTestRaw.csv", encoding="utf-8")
