# This script normalize the filename of RGB images
import os

path = input('please enter file path: ')

fileList = os.listdir(path)

n = 0
for i in fileList:
    oldName = path + os.sep + fileList[n]
    newName = path + os.sep + 'ADL_'+str(n+1)+'.png'
    os.rename(oldName, newName)
    print(oldName, '=====>', newName)
    n += 1
