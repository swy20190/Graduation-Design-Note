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
                for i in range(len(line)):
                    if i % 2 == 0:
                        line[i] = (line[i]-line[2]) / thigh
                    else:
                        line[i] = (line[i]-line[3]) / thigh
            elif mode == "torso_box":
                # min_x, min_y, max_x, max_y of torso
                min_x = min(line[4], line[10], line[16], line[18], line[22], line[24])
                max_x = max(line[4], line[10], line[16], line[18], line[22], line[24])
                min_y = min(line[5], line[11], line[17], line[19], line[23], line[25])
                max_y = max(line[5], line[11], line[17], line[19], line[23], line[25])
                # calculate the normalize index
                x_diff = max_x - min_x
                y_diff = max_y - min_y
                for i in range(len(line)):
                    if i % 2 == 0:
                        line[i] = (line[i]-line[2]) / x_diff
                    else:
                        line[i] = (line[i]-line[3]) / y_diff
            elif mode == "none":
                for i in range(len(line)):
                    if i % 2 == 0:
                        line[i] = line[i] - line[2]
                    else:
                        line[i] = line[i] - line[3]
            dataNormList.append(line)

        dataNormDf = pd.DataFrame(dataNormList, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y',
                                                         '5x', '5y', '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y',
                                                         '10x', '10y', '11x', '11y', '12x', '12y', '13x', '13y'])
        dataNormDf.to_csv(targetName)


normalize("data/ADLPure.csv", "data/normalized/ADLNormAngle.csv", mode="angle")
normalize("data/ADLPure.csv", "data/normalized/ADLNormThigh.csv", mode="thigh_length")
