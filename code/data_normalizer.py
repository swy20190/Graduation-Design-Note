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
    for line in dataList:
        if mode == "thigh_len":
            pass
        elif mode == "torso_box":
            pass
        elif mode == "none":
            pass
        dataNormList.append(line)

    dataNormDf = pd.DataFrame(dataNormList, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x',
                                                     '5y', '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y',
                                                     '11x', '11y', '12x', '12y', '13x', '13y'])
    dataNormDf.to_csv(targetName)
