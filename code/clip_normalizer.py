# This file purify the clip_raw

import pandas as pd
import numpy as np
from math import isnan

clipRawDf = pd.read_csv('data/clipTrainRaw.csv', index_col=0)
clipRawArr = np.array(clipRawDf)
clipRawList = clipRawArr.tolist()
rawLen = len(clipRawList)
print(rawLen)

# begin to purify
clipPureList = []
for i in range(int(rawLen/2)):
    # print(i)
    oddLine = clipRawList[i*2]
    evenLine = clipRawList[i*2+1]
    #  Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6,
    #  Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13,
    #  Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17
    if isnan(oddLine[2]) or (isnan(oddLine[4]) and isnan(oddLine[10])) or (isnan(oddLine[16]) and isnan(oddLine[22])) or \
            (isnan(oddLine[18]) and isnan(oddLine[24])):
        continue
    if isnan(evenLine[2]) or (isnan(oddLine[4]) and isnan(evenLine[10])) or (isnan(oddLine[16]) and isnan(evenLine[22])) or \
            (isnan(evenLine[18]) and isnan(evenLine[24])):
        continue
    if isnan(oddLine[36]) or isnan(evenLine[36]):
        continue
    if isnan(oddLine[4]):
        oddLine[4] = oddLine[10]
        oddLine[5] = oddLine[11]
    if isnan(oddLine[10]):
        oddLine[10] = oddLine[4]
        oddLine[11] = oddLine[5]
    if isnan(oddLine[16]):
        oddLine[16] = oddLine[22]
        oddLine[17] = oddLine[23]
    if isnan(oddLine[22]):
        oddLine[22] = oddLine[16]
        oddLine[23] = oddLine[17]
    if isnan(oddLine[18]):
        oddLine[18] = oddLine[24]
        oddLine[19] = oddLine[25]
    if isnan(oddLine[24]):
        oddLine[24] = oddLine[18]
        oddLine[25] = oddLine[19]

    if isnan(evenLine[4]):
        evenLine[4] = evenLine[10]
        evenLine[5] = evenLine[11]
    if isnan(evenLine[10]):
        evenLine[10] = evenLine[4]
        evenLine[11] = evenLine[5]
    if isnan(evenLine[16]):
        evenLine[16] = evenLine[22]
        evenLine[17] = evenLine[23]
    if isnan(evenLine[22]):
        evenLine[22] = evenLine[16]
        evenLine[23] = evenLine[17]
    if isnan(evenLine[18]):
        evenLine[18] = evenLine[24]
        evenLine[19] = evenLine[25]
    if isnan(evenLine[24]):
        evenLine[24] = evenLine[18]
        evenLine[25] = evenLine[19]

    pureLine = [evenLine[2] - oddLine[2], evenLine[3] - oddLine[3]]
    pureLine.append(evenLine[4] - oddLine[4])
    pureLine.append(evenLine[5] - oddLine[5])
    pureLine.append(evenLine[10] - oddLine[10])
    pureLine.append(evenLine[11] - oddLine[11])
    pureLine.append(evenLine[16] - oddLine[16])
    pureLine.append(evenLine[17] - oddLine[17])
    pureLine.append(evenLine[22] - oddLine[22])
    pureLine.append(evenLine[23] - oddLine[23])
    pureLine.append(evenLine[18] - oddLine[18])
    pureLine.append(evenLine[19] - oddLine[19])
    pureLine.append(evenLine[24] - oddLine[24])
    pureLine.append(evenLine[25] - oddLine[25])

    pureLine.append(oddLine[36])

    clipPureList.append(pureLine)


clipPureDf = pd.DataFrame(clipPureList, columns=['nx', 'ny', 'rsx', 'rsy', 'lsx', 'lsy', 'rhx', 'rhy', 'lhx', 'lhy',
                                                 'rkx', 'rky', 'lkx', 'lky', 'label'])
clipPureDf.to_csv('data/clipPureTrain.csv')


