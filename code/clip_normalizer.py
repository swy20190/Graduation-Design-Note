# This file purify the clip_raw

import pandas as pd
import numpy as np
from math import isnan

def clip_purifier(srcPath, dstPath):
    clipRawDf = pd.read_csv(srcPath)
    clipRawArr = np.array(clipRawDf)
    clipRawList = clipRawArr.tolist()
    rawLen = len(clipRawList)

    clipPureList = []
    for i in range(int(rawLen/2)):
        oddLine = clipRawList[i*2]
        evenLine = clipRawList[i*2 + 1]

        if isnan(oddLine[3]) or (isnan(oddLine[5]) and isnan(oddLine[11])) or (isnan(oddLine[17]) and isnan(oddLine[23])) or \
                (isnan(oddLine[19]) and isnan(oddLine[25])):
            continue
        if isnan(evenLine[3]) or (isnan(evenLine[5]) and isnan(evenLine[11])) or (isnan(evenLine[17]) and isnan(evenLine[23])) or \
                (isnan(evenLine[19]) and isnan(evenLine[25])):
            continue
        if isnan(oddLine[37]) or isnan(evenLine[37]):
            continue

        if isnan(oddLine[5]):
            oddLine[5] = oddLine[11]
            oddLine[6] = oddLine[12]
        if isnan(oddLine[11]):
            oddLine[11] = oddLine[5]
            oddLine[12] = oddLine[6]
        if isnan(oddLine[17]):
            oddLine[17] = oddLine[23]
            oddLine[18] = oddLine[24]
        if isnan(oddLine[23]):
            oddLine[23] = oddLine[17]
            oddLine[24] = oddLine[18]
        if isnan(oddLine[19]):
            oddLine[19] = oddLine[25]
            oddLine[20] = oddLine[26]
        if isnan(oddLine[25]):
            oddLine[25] = oddLine[19]
            oddLine[26] = oddLine[20]

        if isnan(evenLine[5]):
            evenLine[5] = evenLine[11]
            evenLine[6] = evenLine[12]
        if isnan(evenLine[11]):
            evenLine[11] = evenLine[5]
            evenLine[12] = evenLine[6]
        if isnan(evenLine[17]):
            evenLine[17] = evenLine[23]
            evenLine[18] = evenLine[24]
        if isnan(evenLine[23]):
            evenLine[23] = evenLine[17]
            evenLine[24] = evenLine[18]
        if isnan(evenLine[19]):
            evenLine[19] = evenLine[25]
            evenLine[20] = evenLine[26]
        if isnan(evenLine[25]):
            evenLine[25] = evenLine[19]
            evenLine[26] = evenLine[20]

        pureLine = [evenLine[3] - oddLine[3], evenLine[4] - oddLine[4]]
        pureLine.append(evenLine[5] - oddLine[5])
        pureLine.append(evenLine[6] - oddLine[6])
        pureLine.append(evenLine[11] - evenLine[11])
        pureLine.append(evenLine[12] - oddLine[12])
        pureLine.append(evenLine[17] - oddLine[17])
        pureLine.append(evenLine[18] - oddLine[18])
        pureLine.append(evenLine[23] - oddLine[23])
        pureLine.append(evenLine[24] - oddLine[24])
        pureLine.append(evenLine[19] - oddLine[19])
        pureLine.append(evenLine[20] - oddLine[20])
        pureLine.append(evenLine[25] - oddLine[25])
        pureLine.append(evenLine[26] - oddLine[26])

        pureLine.append(oddLine[37])
        pureLine.append(oddLine[0])
        clipPureList.append(pureLine)

    clipPureDf = pd.DataFrame(clipPureList, columns=['nx', 'ny', 'rsx', 'rsy', 'lsx', 'lsy', 'rhx', 'rhy', 'lhx', 'lhy',
                                                     'rkx', 'rky', 'lkx', 'lky', 'label', 'original_index'])
    clipPureDf.dropna(inplace=True)
    clipPureDf.to_csv(dstPath)


clip_purifier('data/clipTrainRaw.csv', 'data/clipTrainPure.csv')
clip_purifier('data/clipTestRaw.csv', 'data/clipTestPure.csv')

