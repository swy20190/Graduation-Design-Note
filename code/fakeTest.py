import pandas as pd
import numpy as np
import cv2

# processing data
# originalDf = pd.read_csv('data/clipTestRaw.csv', index_col=0)
# originalList = np.array(originalDf).tolist()
# fakeList = []
# for i in range(28):
#     fakeList.append(originalList[i])
# for i in range(30, 4912):
#     fakeList.append(originalList[i])
# for i in range(4914, 6018):
#     fakeList.append(originalList[i])
# for i in range(6024, 6502):
#     fakeList.append(originalList[i])
# for i in range(6504, 16726):
#     fakeList.append(originalList[i])
# for i in range(16728, 21474):
#     fakeList.append(originalList[i])
# for i in range(21476, 21868):
#     fakeList.append(originalList[i])
# for i in range(21870, 27440):
#     fakeList.append(originalList[i])
# for i in range(27442, 28294):
#     fakeList.append(originalList[i])
# for i in range(28296, 28298):
#     fakeList.append(originalList[i])
# for i in range(28302, 33872):
#     fakeList.append(originalList[i])
#
# fakeDf = pd.DataFrame(fakeList, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
#                                                '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x',
#                                                '11y', '12x', '12y', '13x', '13y', '14x', '14y', '15x', '15y', '16x',
#                                                '16y', '17x', '17y', 'fall'])
# fakeDf.to_csv('data/clipFake.csv')

# processing frames
ptr = 1
for i in range(1, 27):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(29, 4911):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(4913, 6017):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(6023, 6501):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(6503, 16725):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(16727, 21473):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(21475, 21867):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(21869, 27439):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(27441, 28293):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(28295, 28297):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1

for i in range(28301, 33871):
    src = cv2.imread('clips_test/' + str(i) + '.png')
    cv2.imwrite('clipsFake/' + str(ptr) + '.png', src)
    ptr = ptr+1
