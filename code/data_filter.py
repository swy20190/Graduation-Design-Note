# This file reads the raw csv file and generates filtered purer csv file
import pandas as pd
import numpy as np
from math import isnan


def pure(srcName, targetName):
    """
    read data from src, drop duplicates and fill NAN, store in target
    :param srcName: raw data csv
    :param targetName: pure data csv
    :return: None
    """
    rawDf = pd.read_csv(srcName, index_col=0)
    # drop duplicate lines
    pureDf = rawDf.drop_duplicates(inplace=False)
    # clean NAN row
    pureDf.dropna(how='all')
    # fill nan value
    pureArr = np.array(pureDf)
    pureList = pureArr.tolist()
    #  Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6,
    #  Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13,
    #  Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17
    filledList = []
    for line in pureList:
        if isnan(line[2]) or (isnan(line[4]) and isnan(line[10])) or (isnan(line[16]) and isnan(line[22])) or (isnan(line[18]) and isnan(line[24])):
            pass
        else:
            pureLine = line[0:28]
            if isnan(pureLine[0]):
                pureLine[0] = pureLine[2]
                pureLine[1] = pureLine[3]
            if isnan(pureLine[4]):
                pureLine[4] = pureLine[10]
                pureLine[5] = pureLine[11]
            if isnan(pureLine[10]):
                pureLine[10] = pureLine[4]
                pureLine[11] = pureLine[5]
            if isnan(pureLine[16]):
                pureLine[16] = pureLine[22]
                pureLine[17] = pureLine[23]
            if isnan(pureLine[22]):
                pureLine[22] = pureLine[16]
                pureLine[23] = pureLine[17]
            if isnan(pureLine[18]):
                pureLine[18] = pureLine[24]
                pureLine[19] = pureLine[25]
            if isnan(pureLine[24]):
                pureLine[24] = pureLine[18]
                pureLine[25] = pureLine[19]
        filledList.append(pureLine)

    filledDf = pd.DataFrame(filledList, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
                                                 '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x',
                                                 '11y', '12x', '12y', '13x', '13y'])
    # 使用纵向前值代替nan
    filledDf = filledDf.fillna(axis=0, method='ffill')
    filledDf = filledDf.fillna(value=-999)
    filledDf.drop_duplicates(inplace=True)
    filledDf.to_csv(targetName)


pure("data/ADLRaw.csv", "data/ADLPure.csv")
pure("data/FallRaw.csv", "data/FallPure.csv")
pure("data/ADLRawTest.csv", "data/ADLPureTest.csv")
pure("data/FallRawTest.csv", "data/FallPureTest.csv")
