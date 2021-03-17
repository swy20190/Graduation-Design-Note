# This file adjust the bugged index of clips and clips_test
import os

clipImgList = os.listdir('clips')

imgCnt = len(clipImgList)

for i in range(59813-24095):
    oldName = 'clips\\' + str(i+24096) + '.png'
    newName = 'clips\\' + str(i+24095) + '.png'
    os.rename(oldName, newName)
    print(oldName, '=====>', newName)