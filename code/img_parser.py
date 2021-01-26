# This file parse RGB images into joint coordinates
import cv2
import numpy as np


def parse(net, nPoints, image):
    """
    parse RGB image into joints coordinate data
    :param net: dnn net in memory
    :param nPoints: corresponds to the model
    :param image: RGB file
    :return: a list of (x, y) which is joint coordinate
    """
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

    # Read image
    frame = cv2.imread(image)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1
    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368
    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)
    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        # find global maxima of the probMap
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        # scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        if prob > threshold:
            # add thr point to the list if the probability is greater than the threshold
            points.append((x, y))
        else:
            points.append(None)

    cv2.waitKey(0)
    return points
