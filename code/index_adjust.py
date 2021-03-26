# This file adjust the bugged index of clips and clips_test
import os

clipImgList = os.listdir('clips_test')

imgCnt = len(clipImgList)

for i in range(34405-32188):
    oldName = 'clips_test\\' + str(i+32189) + '.png'
    newName = 'clips_test\\' + str(i+32187) + '.png'
    os.rename(oldName, newName)
    print(oldName, '=====>', newName)