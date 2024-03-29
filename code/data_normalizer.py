# this file normalize the pure data
import pandas as pd
import numpy as np
import math


def normalize(srcName, targetName, mode):
    """
    normalize the pure data
    :param srcName: pure csv name
    :param targetName: norm csv name
    :param mode: the mode of normalize
    :return: None
    """
    dataDf = pd.read_csv(srcName, index_col=0)
    dataArr = np.array(dataDf)
    dataList = dataArr.tolist()
    dataNormList = []
    #  Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6,
    #  Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13,
    #  Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17
    if mode == "angle":
        for line in dataList:
            normal_line = []
            # aspect ratio of torso
            min_x = min(line[2], line[4], line[10], line[16], line[18], line[22], line[24])
            max_x = max(line[2], line[4], line[10], line[16], line[18], line[22], line[24])
            min_y = min(line[3], line[5], line[11], line[17], line[19], line[23], line[25])
            max_y = max(line[3], line[5], line[11], line[17], line[19], line[23], line[25])
            aspect_ratio = (max_x - min_x) / (max_y - min_y)
            normal_line.append(aspect_ratio)
            # angle of knee-neck and the vertical line
            angle_left = math.degrees(math.atan2(line[24]-line[2], line[25]-line[3]))
            angle_right = math.degrees(math.atan2(line[18]-line[2], line[19]-line[3]))
            normal_line.append(angle_left)
            normal_line.append(angle_right)
            dataNormList.append(normal_line)
        dataNormDf = pd.DataFrame(dataNormList, columns=['ratio', 'LAngle', 'RAngle'])
        dataNormDf.to_csv(targetName)
    else:
        for line in dataList:
            if mode == "thigh_len":
                # the length of right thigh
                thigh = math.sqrt((line[16]-line[18])**2 + (line[17]-line[19])**2)
                if thigh == 0:
                    thigh = 0.01
                centerX = line[2]
                centerY = line[3]
                for i in range(len(line)):
                    if i % 2 == 0:
                        line[i] = (line[i]-centerX) / thigh
                    else:
                        line[i] = (line[i]-centerY) / thigh
            elif mode == "torso_box":
                # min_x, min_y, max_x, max_y of torso
                min_x = min(line[4], line[10], line[16], line[18], line[22], line[24])
                max_x = max(line[4], line[10], line[16], line[18], line[22], line[24])
                min_y = min(line[5], line[11], line[17], line[19], line[23], line[25])
                max_y = max(line[5], line[11], line[17], line[19], line[23], line[25])
                centerX = line[2]
                centerY = line[3]
                # calculate the normalize index
                x_diff = max_x - min_x
                y_diff = max_y - min_y
                if x_diff == 0:
                    x_diff = 0.01
                if y_diff == 0:
                    y_diff = 0.01
                for i in range(len(line)):
                    if i % 2 == 0:
                        line[i] = (line[i]-centerX) / x_diff
                    else:
                        line[i] = (line[i]-centerY) / y_diff
            elif mode == "none":
                centerX = line[2]
                centerY = line[3]
                for i in range(len(line)):
                    if i % 2 == 0:
                        line[i] = line[i] - centerX
                    else:
                        line[i] = line[i] - centerY
            dataNormList.append(line)

        dataNormDf = pd.DataFrame(dataNormList, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y',
                                                         '5x', '5y', '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y',
                                                         '10x', '10y', '11x', '11y', '12x', '12y', '13x', '13y'])
        dataNormDf.to_csv(targetName)


normalize("data/ADLPure.csv", "data/normalized/ADLNormAngle.csv", mode="angle")
normalize("data/ADLPure.csv", "data/normalized/ADLNormThigh.csv", mode="thigh_len")
normalize("data/ADLPure.csv", "data/normalized/ADLNormTorso.csv", mode="torso_box")
normalize("data/ADLPure.csv", "data/normalized/ADLNormNone.csv", mode="none")

normalize("data/FallPure.csv", "data/normalized/FallNormAngle.csv", mode="angle")
normalize("data/FallPure.csv", "data/normalized/FallNormThigh.csv", mode="thigh_len")
normalize("data/FallPure.csv", "data/normalized/FallNormTorso.csv", mode="torso_box")
normalize("data/FallPure.csv", "data/normalized/FallNormNone.csv", mode="none")

normalize("data/ADLPureTest.csv", "data/normalized/ADLNormAngleTest.csv", mode="angle")
normalize("data/ADLPureTest.csv", "data/normalized/ADLNormThighTest.csv", mode="thigh_len")
normalize("data/ADLPureTest.csv", "data/normalized/ADLNormTorsoTest.csv", mode="torso_box")
normalize("data/ADLPureTest.csv", "data/normalized/ADLNormNoneTest.csv", mode="none")

normalize("data/FallPureTest.csv", "data/normalized/FallNormAngleTest.csv", mode="angle")
normalize("data/FallPureTest.csv", "data/normalized/FallNormThighTest.csv", mode="thigh_len")
normalize("data/FallPureTest.csv", "data/normalized/FallNormTorsoTest.csv", mode="torso_box")
normalize("data/FallPureTest.csv", "data/normalized/FallNormNoneTest.csv", mode="none")
