# This script normalize the filename of RGB images
import os

ADLPath = input('please enter ADL file path: ')

fileList = os.listdir(ADLPath)

n = 0
for i in fileList:
    oldName = ADLPath + os.sep + fileList[n]
    newName = ADLPath + os.sep + 'ADL_'+str(n+1)+'.png'
    if oldName != newName:
        os.rename(oldName, newName)
    print(oldName, '=====>', newName)
    n += 1

FallPath = input('please enter fall file path: ')

fileList = os.listdir(FallPath)

n = 0
for i in fileList:
    oldName = FallPath + os.sep + fileList[n]
    newName = FallPath + os.sep + "fall_" + str(n+1) + '.png'
    if oldName != newName:
        os.rename(oldName, newName)
    print(oldName, '=====>', newName)
    n += 1
